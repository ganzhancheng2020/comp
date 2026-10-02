# Draft forum question for the organizers (to be posted by the account holder)

**Title:** Task 2: may post-buffer observations (T+95 onwards) of the same day be used?

Hi organizers,

For Task 2, each window's mainline observations are blanked from T+5 to T+90 in `mainline_states_masked`, but the masked
layer publishes the rest of that day, including observations from T+95 onwards. A queue that forms at T+30 is usually
still visible there, so those later observations reveal which links queue in the horizon.

The Task 2 definition says that "at forecast origin T, participants receive the previous 60 minutes", and the rules
prohibit recovering private labels. We read that as: Task 2 predictions may only use data up to T (the window history
and the same-day masked layer before T), and the same-day observations after the buffer must not be used.

Could you confirm whether that reading is correct? Specifically:
1. May a Task 2 prediction for a window use mainline observations of the same day published after T+90?
2. May it use ramp observations inside the T+5…T+90 interval (the ramp layer is not blanked there)?
3. For Task 1 (offline reconstruction), is it fine to use every published observation of the split, including ramp
   observations inside the Task 2 blackouts?

A clear ruling would help everyone compete on the same information set. Thank you!
