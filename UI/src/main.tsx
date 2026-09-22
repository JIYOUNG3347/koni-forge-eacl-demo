
  import { createRoot } from "react-dom/client";
  import App from "./App.tsx";
  import "./index.css";
  import { cleanupLegacyKeys } from "./utils/storage";

  cleanupLegacyKeys();
  createRoot(document.getElementById("root")!).render(<App />);
  