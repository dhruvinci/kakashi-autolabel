You are labelling a 20-second interval of combat-sports video (MMA or no-gi grappling) for a vision dataset. Answer only from the attached images. Do not run commands or read files.

The interval has {n_frames} frames, F00 to F{last:02d}, sampled every ~0.5 s in time order. A local segmentation model has already detected candidate people in every frame. Each candidate is outlined in a colour and tagged with a white-on-colour number (a "mark") placed inside its region. Marks are numbered per frame: the same number in two frames is NOT necessarily the same person.

Attached images:
- {n_keyframes} KEYFRAMES ({keyframe_ids}) at full {width}x{height} resolution, one per image, frame id in the top-left banner.
- The remaining frames as TILE SHEETS: each sheet is a 2x2 grid of four frames at half resolution, frame id in each tile's top-left banner. Read tiles left-to-right, top-to-bottom.

Source metadata (hypothesis only, verify against the images): sport={sport}, context={context}, title="{title}".

Tasks:

1. Identities. Give each distinct non-crowd person a stable actor_id (P1, P2, ...) that stays the same in EVERY frame of the interval, with a short appearance description that would let someone re-identify them (clothing colour, shorts, hair, skin, gloves). Use appearance and continuity across neighbouring frames; camera cuts and replays can change framing abruptly. Assign one role per actor from: {roles}. Interchangeable background spectators/crew/photographers with no stable identity use actor_id "crowd" and are not listed as actors.

2. Marks, for EVERY frame (keyframes and tiles). For each mark return a verdict:
   - "person": one real person (fully or partly visible). Give actor_id (or "crowd") and posture ({postures}).
   - "merged": the mark covers two or more people as one region (common when athletes are entangled). Give the actor_ids in merged_actor_ids.
   - "partial_duplicate": a second mark on a person who already has a better mark in this frame. Give actor_id.
   - "not_a_person": object, shadow, graphic, poster, screen or reflection.
   Judge the outline, not just the tag. Each tag sits inside the region it labels. If a tag is unreadable, omit that mark rather than guess.

3. Unmarked people.
   - KEYFRAMES: list every person (athlete, referee, instructor or coach especially) who does not have their own clean "person" mark, either missed entirely or only covered by a "merged" mark. Entangled athletes inside a merged mark must each be listed. Give a tight pixel box [x1, y1, x2, y2] covering only that person's visible body, in full {width}x{height} coordinates (origin top-left), plus actor_id and posture. Skip crowd members too small or blurred to box individually.
   - TILE FRAMES: the same, but only for non-crowd actors (athletes, referee, instructor, coach). Give the box in the TILE SHEET's own pixel coordinates (the whole {width}x{height} sheet image, origin at the sheet's top-left), tight around that person inside their tile, in box_xyxy_sheet.

4. Frame labels for EVERY frame: shot_type ({shot_types}) and pair_configuration ({pair_configurations}). Pair definition: {pair_definition}

5. Interval label: clip_context ({clip_contexts}).

Return keyframes in "keyframes" and tile frames in "frames", each with its frame id. Use "unknown" rather than guessing. Keep notes to one or two sentences about anything that blocked a confident label.
