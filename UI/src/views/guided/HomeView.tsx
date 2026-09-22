import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { GpuMetricCard, StorageMetricCard } from "../../components/home/SystemMetricCards";
import { useAuthStore } from "../../stores/authStore";
import guidedHeroIllustration from "../../assets/guided-hero-illustration.png";
import {
  Upload, Brain, MessageCircle,
  Bot, CheckCircle2, ChevronRight,
} from "lucide-react";
import { tp, useT } from "../../i18n";

const GUIDED_STEPS = [
  { key: "data",  label: "Upload",   short: "Upload", desc: "Upload PDF/DOCX/PPTX documents or a QA dataset (JSON/JSONL). Documents are indexed for RAG and KBD measures what the model already knows.", path: "/data",  icon: Upload,      agent: "Retrieval · KBD Agent" },
  { key: "train", label: "Training", short: "Train",  desc: "The training agent recommends LoRA / full fine-tuning settings from the dataset size, model size and GPU memory.", path: "/train", icon: Brain,       agent: "Training Agent" },
  { key: "chat",  label: "Chat",     short: "Chat",   desc: "Chat with the fine-tuned model, optionally grounded on the indexed documents.", path: "/chat",  icon: MessageCircle, agent: null },
];

const BRAND = "KONI-Forge";

export default function HomeView() {
  const t = useT();
  const navigate = useNavigate();
  const userId = useAuthStore(s => s.userId);
  const [branded, setBranded] = useState("");
  const typeDone = useRef(false);

  useEffect(() => {
    let i = 0;
    const iv = setInterval(() => {
      if (i < BRAND.length) setBranded(BRAND.slice(0, ++i));
      else { clearInterval(iv); typeDone.current = true; }
    }, 70);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="space-y-5 max-w-[1400px] mx-auto">

      {/* ── Hero — greeting, progress tiles and illustration ─── */}
      <div
        className="relative rounded-[20px] overflow-hidden bg-neutral-950 min-h-[340px] flex items-center px-8 py-7 shadow-[0_8px_40px_rgba(0,0,0,0.25)]"
        style={{ animation: "fade-slide-up 0.5s ease-out both" }}
      >
        {/* animated dot-grid */}
        <div className="pointer-events-none absolute inset-0"
          style={{
            backgroundImage: "radial-gradient(circle, rgba(255,255,255,0.042) 1px, transparent 1px)",
            backgroundSize: "24px 24px",
            animation: "dot-drift 14s linear infinite",
          }} />
        {/* glows */}
        <div className="pointer-events-none absolute -bottom-24 -right-24 w-96 h-96 rounded-full bg-white/[0.028] blur-3xl" />
        <div className="pointer-events-none absolute -top-20 -left-20 w-72 h-72 rounded-full bg-white/[0.015] blur-3xl" />

        {/* Illustration */}
        <div
          className="pointer-events-none absolute right-[5%] top-0 bottom-0 w-[61%] flex items-center justify-center px-4 py-3.5"
          style={{
            maskImage: "linear-gradient(to right, transparent 0%, rgba(0,0,0,0.65) 14%, black 32%)",
            WebkitMaskImage: "linear-gradient(to right, transparent 0%, rgba(0,0,0,0.65) 14%, black 32%)",
          }}
        >
          <img src={guidedHeroIllustration} alt={t("The agent working through the pipeline")}
            className="max-w-full max-h-full w-auto h-auto block opacity-90" />
        </div>

        {/* The text flows into the illustration; centre it vertically in the gap. */}
        <div
          className="pointer-events-none absolute top-1/2 -translate-y-1/2 left-[33%] w-[17%] h-[3px]"
          style={{
            backgroundImage: "radial-gradient(circle, rgba(255,255,255,0.20) 1px, transparent 1px)",
            backgroundSize: "7px 3px",
            backgroundRepeat: "repeat-x",
          }}
        >
          {[0, 1.2, 2.4].map((delay) => (
            <i
              key={delay}
              className="absolute -top-px w-[5px] h-[5px] rounded-full bg-white/80 opacity-0"
              style={{
                animation: "conn-travel 3.6s cubic-bezier(0.22,0.61,0.36,1) infinite",
                animationDelay: `${delay}s`,
              }}
            />
          ))}
        </div>

        <div className="relative w-full flex items-start justify-between gap-6">
          <div className="flex-1">
            {/* Mode badge row */}
            <div className="flex items-center gap-3 mb-[18px]">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/[0.1] border border-white/[0.15] text-[11px] font-black text-white uppercase tracking-[0.14em]">
                <span className="w-1.5 h-1.5 rounded-full bg-white/70 flex-shrink-0"
                  style={{ animation: "live-dot 2.4s ease-in-out infinite" }} />
                GUIDED
              </span>
              <span className="text-[10px] font-bold text-neutral-500 uppercase tracking-[0.18em]">
                {branded}
                {!typeDone.current && (
                  <span className="inline-block w-px h-3 bg-neutral-400 ml-0.5 align-middle"
                    style={{ animation: "cursor-blink 0.6s ease infinite" }} />
                )}
              </span>
            </div>

            {/* Greeting */}
            <h1 className="text-[36px] font-black text-white leading-[1.12] tracking-[-0.035em]">
              {t("Hello,")}
              <br />
              {userId ? tp("{0}", userId) : "Welcome"}
            </h1>

          </div>
        </div>

        {/* Attribution */}
        <p className="absolute bottom-4 right-6 text-[12px] text-white/80 tracking-wide select-none">
          {t("KISTI")} <span className="mx-0.5 opacity-50">|</span> {t("Large-scale AI Research Center")}
        </p>
      </div>

      {/* ── System metrics — GPU aggregate and devices / storage by purpose ─── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="h-full" style={{ animation: "fade-slide-up 0.45s ease-out both", animationDelay: "0.1s" }}>
          <GpuMetricCard />
        </div>
        <div className="h-full" style={{ animation: "fade-slide-up 0.45s ease-out both", animationDelay: "0.17s" }}>
          <StorageMetricCard />
        </div>
      </div>

      {/* ── Pipeline steps ────────────────────────────────────────── */}
      <div className="bg-white border border-neutral-200/80 rounded-[20px] overflow-hidden shadow-[0_1px_3px_rgba(0,0,0,0.04)]"
        style={{ animation: "fade-slide-up 0.45s ease-out both", animationDelay: "0.24s" }}>
        <div className="flex items-center justify-between px-5 pt-[18px] pb-3.5">
          <span className="text-[13px] font-extrabold tracking-[-0.02em] text-neutral-900">{t("Pipeline stages")}</span>
        </div>

        <div className="px-3 pb-3 grid grid-cols-1 gap-2">
          {GUIDED_STEPS.map((step, i) => {
            const isDone    = false;
            const isRunning = false;
            const Icon = step.icon;
            return (
              <button key={step.key} onClick={() => navigate(step.path)}
                className="group w-full text-left"
                style={{ animation: "fade-slide-up 0.35s ease-out both", animationDelay: `${0.28 + i * 0.06}s` }}>
                <div className={`relative flex items-center gap-3.5 px-4 py-[13px] rounded-[14px] transition-all duration-200 ${
                  isRunning
                    ? "bg-white shadow-[0_2px_10px_rgba(0,0,0,0.09)]"
                    : "bg-neutral-50 hover:bg-white hover:shadow-[0_2px_10px_rgba(0,0,0,0.07)]"
                }`}>
                  {isRunning && (
                    <div className="absolute left-0 top-2.5 bottom-2.5 w-[3px] rounded-full bg-neutral-900" />
                  )}
                  <div className={`w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-[10px] font-black transition-all duration-200 ${
                    isRunning ? "bg-neutral-900 text-white"
                    : "bg-neutral-100 text-neutral-400 group-hover:bg-neutral-900 group-hover:text-white"
                  }`}>
                    {isDone ? <CheckCircle2 className="w-3 h-3 text-neutral-400" /> : i + 1}
                  </div>
                  <div className="w-9 h-9 rounded-xl flex-shrink-0 flex items-center justify-center bg-white border border-neutral-200/70 transition-all duration-200 group-hover:bg-neutral-900 group-hover:border-neutral-900">
                    <Icon className={`w-4 h-4 transition-colors duration-200 group-hover:text-white ${
                      isRunning ? "text-neutral-900" : "text-neutral-400"
                    }`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`text-[12.5px] font-extrabold tracking-[-0.02em] ${isDone ? "text-neutral-500" : "text-neutral-900"}`}>
                        {t(step.label)}
                      </span>
                      {step.agent && (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[9px] font-semibold bg-neutral-100 text-neutral-400">
                          <Bot className="w-2.5 h-2.5" />
                          {step.agent}
                        </span>
                      )}
                    </div>
                    <p className={`text-[10px] mt-0.5 truncate ${isDone ? "text-neutral-300" : isRunning ? "text-neutral-500" : "text-neutral-400"}`}>
                      {t(step.desc)}
                    </p>
                  </div>
                  {isRunning ? (
                    <span className="flex-shrink-0 flex items-center gap-1.5 text-[9px] font-bold text-white bg-neutral-900 px-[11px] py-[5px] rounded-full">
                      <span className="w-[5px] h-[5px] rounded-full bg-white animate-pulse" />
                      {t("In progress")}
                    </span>
                  ) : isDone ? (
                    <span className="flex-shrink-0 text-[9px] font-semibold text-neutral-300 px-2 py-1">{t("Completed")}</span>
                  ) : (
                    <ChevronRight className="w-4 h-4 flex-shrink-0 text-neutral-300 group-hover:text-neutral-600 group-hover:translate-x-0.5 transition-all duration-200" />
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

    </div>
  );
}
