import { useEffect, useState } from "react";

type Theme = "light" | "dark";
const THEME_KEY = "minimal-theme"; // same key as jkmr and the demo page

/** Light/dark on <body class="minimal" data-theme>, remembered in localStorage. */
export function useTheme(): [Theme, () => void] {
  const [t, setT] = useState<Theme>(() => {
    try {
      const s = localStorage.getItem(THEME_KEY);
      if (s === "light" || s === "dark") return s;
    } catch {
      /* ignore */
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.body.dataset.theme = t;
  }, [t]);
  return [
    t,
    () =>
      setT((x) => {
        const n = x === "dark" ? "light" : "dark";
        try {
          localStorage.setItem(THEME_KEY, n);
        } catch {
          /* ignore */
        }
        return n;
      }),
  ];
}
