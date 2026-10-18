// A ten-line router: the demo has two kinds of page (the live demo at /, the
// Foundations track under /learn), so pushState + one hook is all it needs.

import { useSyncExternalStore, type MouseEvent } from "react";

const subs = new Set<() => void>();
window.addEventListener("popstate", () => subs.forEach((f) => f()));

export function usePath(): string {
  return useSyncExternalStore(
    (f) => {
      subs.add(f);
      return () => subs.delete(f);
    },
    () => location.pathname,
  );
}

export function navigate(to: string) {
  if (to === location.pathname) return;
  history.pushState(null, "", to);
  subs.forEach((f) => f());
  window.scrollTo({ top: 0 });
}

/** onClick for <a href>: client-side navigation, but keep cmd/ctrl-click working. */
export function linkTo(to: string) {
  return (e: MouseEvent) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    navigate(to);
  };
}
