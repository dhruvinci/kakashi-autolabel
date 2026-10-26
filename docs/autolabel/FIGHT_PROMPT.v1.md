You are labelling the fight state of a 20-second interval of combat-sports video (MMA, no-gi or gi grappling) for a training dataset. Answer only from the attached images. Do not run commands or read files.

The interval has {n_frames} frames, F00 to F{last:02d}, about every 0.5 s, shown as 2x2 TILE SHEETS (read tiles left-to-right, top-to-bottom; each tile has its frame id top-left). The tracked people are outlined and tagged with their actor id in a fixed colour on every frame:
{actor_lines}
Untagged people are background. Source metadata (hypothesis only): sport={sport}, context={context}, title="{title}". Interval clip context from a previous pass: {clip_context}.

For EVERY frame return:
- engagement: idle | engaged | scramble | not_visible (L0). "engaged" when the two athletes are in a recognisable exchange or position.
- position (L1A): standing | guard | side_control | mount | back | turtle | leg_entanglement | not_applicable (no two athletes interacting) | unknown.
- position_detail (L1B), one term for the position, from: {l1b}; or "none"/"unknown".
- position_alt: your second most likely L1A for this frame (same vocabulary), and confidence: high | medium | low for the first choice.
- top_actor: actor id of the athlete on top / in the dominant pinning position, or null when standing or neither.
- dominant_actor: actor id of the athlete currently controlling or attacking, "neutral", or "unknown".

Then list EVENTS over the interval (may be empty), each with type, actor (who does it), target, start_frame and end_frame ids, outcome (success | failed | ongoing | unknown) and detail:
- grappling transitions (Kakashi T): takedown, sweep, pass, back_take, mount_achieve, escape, reguard, leg_entry, reset, stand_up;
- submission_attempt (detail = rear_naked_choke, guillotine, darce, anaconda, arm_triangle, triangle, armbar, kimura, americana, omoplata, heel_hook, kneebar, straight_ankle_lock, toe_hold, calf_slicer, other) and tap;
- MMA striking: striking_exchange (a period of strikes being thrown, detail = standing or ground), knockdown, referee_stoppage. Do not try to classify individual punches/kicks at this frame rate.

Disambiguation rules (most common errors - check them):
- Half guard vs side control: ANY bottom-player leg entanglement of the top player's leg means half guard (or quarter guard), not side control. Side control needs both bottom legs completely free.
- Mount vs back: top player facing the same direction as the bottom player's face (chest to chest/front) = mount; chest to the defender's back = back.
- Turtle vs scramble: one athlete stable on hands-and-knees or belly-down = turtle; both moving with no settled base for > ~2 s = scramble. Do not use scramble as a catch-all for long periods.
- Closed vs open guard: closed requires ankles crossed behind the top player's back.
- Guard requires the bottom player's legs between the athletes; back requires chest-to-back; mount requires straddling the torso.
- Do not invent grips, submissions or events you cannot see; use unknown and low confidence instead. Transitions are labelled at completion, with start_frame where the attempt visibly begins.

Return frames in order with their frame ids. Keep notes to one or two sentences.
