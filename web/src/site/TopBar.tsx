import type { ReactNode } from "react";
import { linkTo } from "../learn/router";
import { useTheme } from "../learn/theme";

/** The bar at the top of every page: a skip link, brand, where you are, course / lab links, theme.
 *  Every page marks its main content with id="content" for the skip link. */
export function TopBar({ crumb, here }: { crumb?: ReactNode; here: "home" | "learn" | "lab" }) {
  const [theme, toggle] = useTheme();
  return (
    <div className="nv-wide nv-top lr-top">
      <a className="nv-skip" href="#content">
        Skip to content
      </a>
      <span className="nv-brand">
        <a href="/" onClick={linkTo("/")} className="lr-home">
          The Climb
        </a>
        {crumb && <span className="lr-crumb">{crumb}</span>}
      </span>
      <span className="lr-nav">
        <a href="/learn" onClick={linkTo("/learn")} className={here === "learn" ? "on" : ""}>
          course
        </a>
        <a href="/lab" onClick={linkTo("/lab")} className={here === "lab" ? "on" : ""}>
          lab
        </a>
        <a href="https://github.com/joshuaramkissoon/nanoinfer" className="lr-hide-sm">
          source
        </a>
        <button className="nv-iconbtn" onClick={toggle} aria-label={theme === "dark" ? "switch to light theme" : "switch to dark theme"}>
          {theme === "dark" ? "☀" : "☾"}
        </button>
      </span>
    </div>
  );
}
