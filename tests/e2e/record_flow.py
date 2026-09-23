"""Browser end-to-end test of the full mock-session workflow.

Uses Chromium's synthetic camera/microphone (--use-fake-device-for-media-stream), so the real
getUserMedia -> MediaRecorder -> upload path runs without hardware. Requires the API on :8000 and
the built frontend served by `npm run preview` on :4173.

    python tests/e2e/record_flow.py [--screens docs/screenshots]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = "http://127.0.0.1:4173"
OUT = Path(sys.argv[sys.argv.index("--screens") + 1]) if "--screens" in sys.argv else None


def shot(page, name: str) -> None:  # type: ignore[no-untyped-def]
    if OUT:
        OUT.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
            ]
        )
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, permissions=["camera", "microphone"])
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(BASE + "/")
        page.get_by_placeholder("Your name").fill("E2E operator")
        expect(page.get_by_role("heading", name="Overview")).to_be_visible()

        # register participant + consent
        page.goto(BASE + "/participants")
        page.get_by_role("button", name="Register participant").first.click()
        page.get_by_text("18 or older").click()
        page.get_by_role("dialog").get_by_role("button", name="Register", exact=True).click()
        expect(page.get_by_role("dialog")).to_contain_text("Consent for P-")
        for box in page.get_by_role("dialog").locator("input[type=checkbox]").all():
            box.check()
        page.get_by_label("Participant's full name").fill("Test Volunteer")
        page.get_by_label("Ethics approval reference").fill("ETH-DEMO-1")
        shot(page, "02-consent")
        page.get_by_role("dialog").get_by_role("button", name="Record consent").click()
        expect(page.get_by_text("Given (consent-v1.0)").first).to_be_visible()
        shot(page, "03-participants")

        # new session with the cue rehearsal script
        page.goto(BASE + "/sessions/new")
        page.locator("#participant").select_option(index=1)
        page.get_by_text("Cue rehearsal (not part of the corpus)").click()
        page.get_by_role("button", name="Dim").click()
        page.get_by_role("button", name="Standard (480p)").click()
        shot(page, "04-new-session")
        page.get_by_role("button", name="Create session").click()
        page.get_by_text("agrees to continue today").click()
        page.get_by_role("button", name="Confirm and continue").click()
        page.wait_for_url("**/record")

        # equipment check
        page.get_by_role("button", name="Turn on camera and microphone").click()
        expect(page.get_by_text("Microphone working")).to_be_visible(timeout=15000)
        page.get_by_role("button", name="Take photo").click()
        expect(page.get_by_text("Enrolment photo saved").first).to_be_visible(timeout=10000)
        shot(page, "05-equipment-check")
        page.get_by_role("button", name="Start recording").click()
        expect(page.get_by_role("heading", name="Recording")).to_be_visible()

        # participant activity: typing and a paste
        page.locator("#q1").fill("Weather is short term, climate is the long-term pattern.")
        page.locator("#q2").press_sequentially("3 hours 35 minutes", delay=20)
        page.evaluate(
            """() => { const dt = new DataTransfer(); dt.setData('text/plain', 'pasted answer text');
            const ev = new ClipboardEvent('paste', {clipboardData: dt, bubbles: true});
            document.getElementById('q3').dispatchEvent(ev); }"""
        )
        expect(page.get_by_role("alert").filter(has_text="Practice cue")).to_be_visible(timeout=30000)
        shot(page, "06-recording-cue")
        time.sleep(17)  # let the cue finish (20 s + 15 s)
        page.get_by_role("button", name="Stop recording").click()
        page.get_by_role("dialog").get_by_role("button", name="Stop recording").click()
        expect(page.get_by_role("heading", name="Review and upload")).to_be_visible(timeout=20000)
        shot(page, "07-review")
        page.get_by_role("button", name="Upload recording").click()
        page.wait_for_url(lambda u: u.rstrip("/").split("/")[-1].startswith("sess_"), timeout=60000)
        expect(page.get_by_text("Uploaded").first).to_be_visible()
        expect(page.get_by_text("Rejected (dead letter)")).to_be_visible()
        sid = page.url.rstrip("/").split("/")[-1]
        page.wait_for_timeout(1500)
        shot(page, "08-session-detail")

        # verify through the API what the browser produced
        detail = page.request.get(f"{BASE}/api/sessions/{sid}").json()
        assert detail["status"] == "UPLOADED", detail["status"]
        rec = next(r for r in detail["recordings"] if r["kind"] == "webcam_av")
        assert rec["size_bytes"] > 10_000, rec
        assert detail["episodes"][0]["shown_at_ms"] is not None
        assert detail["label"]["source"] == "mock" and detail["label"]["confidence"] == "certain"
        counts = detail["event_counts"]
        assert counts.get("screen", 0) >= 3 and counts.get("behavioral", 0) >= 3 and counts.get("device", 0) >= 1, (
            counts
        )
        events = page.request.get(f"{BASE}/api/sessions/{sid}/events?channel=screen").json()
        assert any(e["event_type"] == "PASTE" and e["payload"]["length"] == len("pasted answer text") for e in events)
        video = page.request.get(f"{BASE}/api/recordings/{rec['id']}/content")
        assert video.status == 200 and video.body()[:4] == b"\x1a\x45\xdf\xa3"

        # Phase 2-5 from the UI: analyse the uploaded recording, then record a reviewer decision
        page.goto(BASE + f"/sessions/{sid}")
        page.get_by_role("button", name="Analyse recording").click()
        expect(page.get_by_text("Reviewer decision")).to_be_visible(timeout=120000)
        page.get_by_role("button", name="No concern").click()
        page.get_by_role("button", name="Save decision").click()
        expect(page.get_by_text("Review saved.")).to_be_visible()
        page.wait_for_timeout(800)
        shot(page, "08b-session-analysed")
        analysed = page.request.get(f"{BASE}/api/sessions/{sid}").json()
        assert analysed["status"] == "REVIEWED", analysed["status"]
        assert analysed["event_counts"].get("presence", 0) > 0  # detector events from the real video

        page.goto(BASE + "/review")
        expect(page.get_by_role("heading", name="Review queue")).to_be_visible()
        page.wait_for_timeout(800)
        shot(page, "14-review-queue")
        page.locator("tbody tr").first.click()
        expect(page.get_by_text("Reviewer decision")).to_be_visible(timeout=20000)
        page.wait_for_timeout(800)
        shot(page, "15-flagged-session")
        page.goto(BASE + "/model")
        expect(page.get_by_text("What drives the score")).to_be_visible(timeout=20000)
        page.wait_for_timeout(800)
        shot(page, "16-model")
        page.goto(BASE + "/fairness")
        expect(page.get_by_text("Drift monitor")).to_be_visible(timeout=20000)
        page.wait_for_timeout(800)
        shot(page, "17-fairness")

        # remaining screens
        page.goto(BASE + "/")
        page.wait_for_timeout(800)
        shot(page, "01-overview")
        page.goto(BASE + "/sessions?source=SIMULATED&violation=true")
        page.wait_for_timeout(800)
        shot(page, "09-simulated-sessions")
        page.locator("tbody tr").first.click()
        expect(page.get_by_text("Phone detected").or_(page.get_by_text("Gaze off screen")).first).to_be_visible(
            timeout=20000
        )
        page.wait_for_timeout(500)
        shot(page, "10-simulated-detail")
        page.goto(BASE + "/datasets")
        expect(page.get_by_text("Behaviour profiles")).to_be_visible()
        page.wait_for_timeout(800)
        shot(page, "11-datasets")
        page.goto(BASE + "/system")
        page.wait_for_timeout(800)
        shot(page, "12-system")

        mobile = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True)
        m = mobile.new_page()
        m.goto(BASE + f"/sessions/{sid}")
        m.wait_for_timeout(1200)
        shot(m, "13-mobile-session")
        assert not errors, errors
        print(f"E2E OK: session {sid}, video {rec['size_bytes']} bytes, events {counts}")
        browser.close()


if __name__ == "__main__":
    main()
