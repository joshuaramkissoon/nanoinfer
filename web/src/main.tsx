import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import Landing from "./site/Landing";
import LearnApp from "./learn/LearnApp";
import { usePath } from "./learn/router";
import { AskLayer } from "./ask/AskLayer";
import "./styles.css";

/** Three kinds of page: the landing page, the course (/learn/...), the lab (/lab). */
function Root() {
  const path = usePath();
  const page = path.startsWith("/learn") ? <LearnApp /> : path.startsWith("/lab") ? <App /> : <Landing />;
  return (
    <>
      {page}
      <AskLayer />
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
