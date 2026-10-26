You are labelling a 20-second interval of combat-sports video (MMA or no-gi grappling) for a vision dataset. Answer only from the attached images. Do not run commands or read files.

The attached images are {n_keyframes} keyframes from the interval, in time order, named in their top banner (K1, K2, ...) with their timestamp. A local segmentation model has already detected candidate people: each candidate is outlined in a colour and tagged with a white-on-colour number ("mark"). Marks are numbered per keyframe; the same number in two keyframes is NOT necessarily the same person.

Source metadata (hypothesis only, verify against the images): sport={sport}, context={context}, title="{title}".

Tasks:

1. Identities. Give each distinct non-crowd person a stable actor_id (P1, P2, ...) that stays the same across all keyframes, with a short appearance description that would let someone re-identify them (clothing colour, shorts, hair, skin, gloves). Assign one role per actor from: {roles}. Interchangeable background spectators/crew/photographers with no stable identity use actor_id "crowd" and are not listed as actors.

2. Marks. For every mark in every keyframe, return a verdict:
   - "person": one real person (fully or partly visible). Give actor_id and posture ({postures}).
   - "merged": the mark covers two or more people as one region (common when athletes are entangled). Give the actor_ids involved in merged_actor_ids.
   - "partial_duplicate": a second mark on a person who already has a better mark in this keyframe.
   - "not_a_person": object, shadow, graphic, poster, screen or reflection.
   Judge the outline, not just the number tag position. Each tag sits inside the region it labels.

3. Unmarked people. For each keyframe, list every person (athlete, referee, instructor or coach especially) who does not have their own clean "person" mark: either missed entirely, or only covered by a "merged" mark. Entangled athletes inside a merged mark must each be listed here. Give a tight pixel box covering only that person's visible body [x1, y1, x2, y2] in the {width}x{height} image coordinates (origin top-left), actor_id, and posture. Do not list crowd members that are too small or blurred to box individually.

4. Frame labels for each keyframe: shot_type ({shot_types}) and pair_configuration ({pair_configurations}). Definitions: {pair_definition}

5. Interval label: clip_context ({clip_contexts}).

Use "unknown" rather than guessing. Keep notes to one or two sentences about anything that blocked a confident label.
