You are choosing which 20-second window of a combat-sports video to label for a training dataset. Answer only from the attached image. Do not run commands or read files.

The image is an overview sheet of 48 thumbnails sampled across the whole video (8 per row, time order, each tagged with its time in seconds). Video: "{title}", {duration} s long. Metadata hypotheses (verify visually): sport={sport}, context={context}.

1. scope: what the footage actually is: mma (striking and grappling with gloves), nogi_grappling (jiu-jitsu/submission grappling without gi), gi_grappling (kimono/gi jiu-jitsu or judo-style gi), other_sport, or not_fight_footage (talking head, podcast, slideshow, mostly graphics).
2. picks: two ranked start times (seconds) for {window}-second windows that contain active engagement between at least two people (sparring, rolling, competing, or an instructor demonstrating on a partner). Prefer, in order: clear live action over talking or intros; variety (if the video contains both standing and ground work, the two picks should cover different phases); unobstructed views; avoid replays, scoreboard graphics, crowd shots, title cards and the last seconds of the video. Each start must leave {window} seconds before the end. Give content type and a one-line reason.

If nothing qualifies, return an empty picks list and explain in notes.
