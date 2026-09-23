# Mock-session protocol (FR-2, DR-2)

## Before anyone is recorded

1. Obtain ethics approval if your institution requires it (ETH-7). Record the reference number; it goes on each consent record.
2. Use Chrome or Edge (best telemetry coverage) or Firefox. Safari cannot record WebM and is refused.
3. Run the app on `http://localhost` or over HTTPS. Browsers block the camera on plain-HTTP remote addresses.

## Each participant

1. **Participants → Register participant.** Confirm they are 18 or older. Do not type their name anywhere except the consent form.
2. The participant reads the consent form, ticks all four items and types their name.
3. For second-person, remote-assistance and impersonation scripts, the helper is registered and consents the same way.

## Each session

1. **New session.** Pick the participant, the script and the conditions as they really are: lighting, webcam class, noise, glasses, head covering. These become the fairness slices, so do not guess.
2. Confirm consent for today.
3. **Check equipment.**
   - Turn on the camera and microphone and choose the right devices.
   - Ask the participant to speak until the meter shows it is working.
   - Take the enrolment photo of the **registered** participant.
4. **Start recording.** The participant answers the quiz. When a cue appears, they follow it until it disappears. Cue times are fixed by the script, and the ground-truth label is built from the schedule.
5. Record for **20–30 minutes** (sessions under 20 minutes are stored but do not count toward the corpus). Stop, review the playback, and upload before closing the tab.

A rehearsal script exists to test equipment; rehearsals never count.

## Coverage target

- ≥ 60 eligible sessions.
- All three lighting conditions (bright, normal, dim) and at least two webcam classes.
- The Datasets page shows the grid and progress.
- A reasonable mix: about 40% clean, and the remainder spread across the six violation scripts.

## Withdrawal

Open the participant and choose **Withdraw participant**. Their recordings, telemetry, labels and optional details
are deleted immediately, including sessions where they were a helper. The consent record remains with the name
removed.
