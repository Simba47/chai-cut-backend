/**
 * Every kind of transition the database takes (transitions.type): 'cut' (none) and every kind the
 * export draws (python/render.py _XFADE). The editor offers them (its TransitionsPanel) and saves
 * one the database doesn't take as a plain dissolve — so a kind left out here quietly exported as
 * a dissolve (user, 2026-11-04: Flare, Glitch, Shake, Spin, Swipe up, Velocity, Warp, Zoom out).
 * transitionKinds.test.ts checks this list against render.py.
 */
export const TRANSITION_KINDS = [
  'cut', 'fade', 'wipe', 'fadeblack', 'fadewhite', 'slideleft', 'slideup', 'zoomin', 'circleopen',
  'slideright', 'slidedown', 'smoothleft', 'hblur', 'radial', 'pixelize',
  'swipeup', 'velocity', 'zoomout', 'shake', 'glitch', 'flare', 'spin', 'warp',
] as const
