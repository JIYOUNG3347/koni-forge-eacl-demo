/**
 * Landing (login) screen.
 *
 * Left: brand panel (dark, same treatment as the home hero — dot grid, glow, illustration).
 * Right: the login form. The auth flow is unchanged (POST /api/auth/login, then localStorage).
 * Below lg the left panel is hidden and the form is centred.
 */
import React, { useState, useEffect } from "react";
import { Eye, EyeOff, Loader2, ArrowRight } from "lucide-react";
import { KoniForgeIcon } from "./icons/KoniForgeIcon";
import landingIllustration from "../assets/landing-illustration.png";
import { useT } from "../i18n";

interface LoginPageProps {
  onLogin: (userId: string, role: string) => void;
}

const rise = (delay: string): React.CSSProperties => ({
  animation: "fade-slide-up 0.7s cubic-bezier(0.22,0.61,0.36,1) both",
  animationDelay: delay,
});

export function LoginPage({ onLogin }: LoginPageProps) {
  const t = useT();
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setReady(true), 60);
    return () => clearTimeout(t);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userId.trim() || !password) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: userId.trim(), password }),
      });
      if (res.ok) {
        const data = await res.json();
        localStorage.setItem("kf-token", data.token);
        localStorage.setItem("kf-user-id", data.user_id);
        localStorage.setItem("kf-user-role", data.role);
        onLogin(data.user_id, data.role);
      } else {
        const err = await res.json().catch(() => ({}));
        setError(err.detail || t("Incorrect ID or password."));
      }
    } catch {
      setError(t("Cannot reach the server."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex bg-white">

      {/* ── Left: brand panel ───────────────────────────────── */}
      <div className="relative hidden lg:flex w-[57%] flex-shrink-0 flex-col overflow-hidden bg-neutral-950 px-10 pt-16 pb-6">
        {/* dot-grid */}
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            backgroundImage: "radial-gradient(circle, rgba(255,255,255,0.042) 1px, transparent 1px)",
            backgroundSize: "24px 24px",
            animation: "dot-drift 14s linear infinite",
          }}
        />
        {/* glows */}
        <div className="pointer-events-none absolute -top-32 -right-32 w-[420px] h-[420px] rounded-full bg-white/[0.03] blur-3xl" />
        <div className="pointer-events-none absolute -bottom-24 -left-24 w-[340px] h-[340px] rounded-full bg-white/[0.016] blur-3xl" />

        {/* Brand */}
        <div className="relative flex items-center gap-3 mb-9" style={rise("0.10s")}>
          <KoniForgeIcon size={32} strokeWidth={1.6} className="text-white/90" />
          <span className="text-[10px] font-bold text-neutral-500 uppercase tracking-[0.18em]">
            KONI-Forge
          </span>
        </div>

        {/* Headline */}
        <h1
          className="relative text-[38px] xl:text-[48px] font-black text-white leading-[1.22] tracking-[-0.03em]"
          style={rise("0.18s")}
        >{t("Work you used to do step by step")}<br />
          <span className="font-normal text-neutral-600">{t("now automated by multiple agents")}</span>
        </h1>

        {/* Illustration */}
        <div
          className="relative flex-1 flex items-end min-h-0 mt-4 -mx-10 -mb-3"
          style={rise("0.30s")}
        >
          <img
            src={landingIllustration}
            alt={t("A team working through an LLMOps pipeline")}
            className="w-full h-auto block opacity-95"
          />
        </div>

        {/* Institution */}
        <p className="absolute bottom-4 right-6 text-[12px] text-white/80 tracking-wide select-none">
          KISTI <span className="mx-0.5 opacity-50">|</span>{t("Large-scale AI Research Center")}</p>
      </div>

      {/* ── Right: login form ───────────────────────────────── */}
      <div className="flex-1 flex items-center justify-center p-10">
        <div
          className="w-full max-w-[340px] transition-opacity duration-500"
          style={{ opacity: ready ? 1 : 0 }}
        >
          {/* Brand mark, shown only on narrow screens */}
          <div className="lg:hidden flex items-center gap-2.5 mb-8">
            <KoniForgeIcon size={28} strokeWidth={1.6} className="text-neutral-800" />
            <span className="text-[10px] font-bold text-neutral-400 uppercase tracking-[0.18em]">
              KONI-Forge
            </span>
          </div>

          <h2 className="text-[26px] font-extrabold tracking-[-0.035em] text-neutral-900" style={rise("0.62s")}>{t("Welcome")}</h2>
          <p className="text-[13.5px] text-neutral-400 mt-2 mb-7 leading-relaxed" style={rise("0.70s")}>{t("Sign in with the account you were issued.")}</p>

          <form onSubmit={handleSubmit}>
            <div className="mb-4" style={rise("0.78s")}>
              <label
                htmlFor="user-id"
                className="block font-mono text-[10px] tracking-[0.13em] uppercase text-neutral-400 mb-2"
              >{t("Account")}</label>
              <input
                id="user-id"
                type="text"
                value={userId}
                onChange={(e) => { setUserId(e.target.value); if (error) setError(""); }}
                placeholder={t("User ID")}
                autoFocus
                autoComplete="username"
                spellCheck={false}
                className="w-full h-12 px-3.5 rounded-[9px] border border-neutral-200 bg-white text-[14.5px] text-neutral-800 placeholder:text-neutral-300 outline-none hover:border-neutral-300 focus:border-neutral-700 focus:ring-[3.5px] focus:ring-neutral-800/[0.07] transition-all"
              />
            </div>

            <div className="mb-4" style={rise("0.86s")}>
              <label
                htmlFor="password"
                className="block font-mono text-[10px] tracking-[0.13em] uppercase text-neutral-400 mb-2"
              >{t("Password")}</label>
              <div className="relative">
                <input
                  id="password"
                  type={showPw ? "text" : "password"}
                  value={password}
                  onChange={(e) => { setPassword(e.target.value); if (error) setError(""); }}
                  placeholder={t("Password")}
                  autoComplete="current-password"
                  className={`w-full h-12 pl-3.5 pr-11 rounded-[9px] border border-neutral-200 bg-white text-neutral-800 placeholder:text-neutral-300 outline-none hover:border-neutral-300 focus:border-neutral-700 focus:ring-[3.5px] focus:ring-neutral-800/[0.07] transition-all ${
                    showPw || !password ? "text-[14.5px]" : "text-[16px] tracking-[0.22em]"
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  aria-label={showPw ? t("Hide password") : t("Show password")}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 w-[30px] h-[30px] flex items-center justify-center rounded-[7px] text-neutral-300 hover:text-neutral-600 hover:bg-neutral-50 transition-colors"
                  tabIndex={-1}
                >
                  {showPw ? <EyeOff className="w-[15px] h-[15px]" /> : <Eye className="w-[15px] h-[15px]" />}
                </button>
              </div>
            </div>

            {error && (
              <div className="text-[12px] text-red-500 font-medium bg-red-50 border border-red-100 rounded-lg px-3 py-2 mb-3">
                {t(error)}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !userId.trim() || !password}
              className="group w-full h-12 mt-2 rounded-[9px] bg-neutral-900 text-white text-[14px] font-bold tracking-[-0.01em] flex items-center justify-center gap-2 shadow-[inset_0_1px_0_rgba(255,255,255,0.16)] hover:-translate-y-px hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.2),0_8px_20px_-8px_rgba(0,0,0,0.6)] active:translate-y-0 disabled:opacity-30 disabled:cursor-not-allowed disabled:translate-y-0 disabled:shadow-none transition-all"
              style={rise("0.94s")}
            >
              {loading ? (
                <><Loader2 className="w-4 h-4 animate-spin" /><span>{t("Preparing agents")}</span></>
              ) : (
                <>
                  <span>{t("Sign in")}</span>
                  <ArrowRight className="w-3.5 h-3.5 opacity-50 group-hover:translate-x-[3px] transition-transform" />
                </>
              )}
            </button>
          </form>

          <p className="text-[12.5px] text-neutral-400 text-center mt-6" style={rise("1.02s")}>{t("No account? ")}<span className="text-neutral-600 border-b border-neutral-200">{t("Ask an administrator")}</span>{t("")}</p>

          <div
            className="flex items-center justify-center gap-2 mt-6 font-mono text-[10px] tracking-[0.08em] uppercase text-neutral-300"
            style={rise("1.10s")}
          >
            <span
              className="w-[5px] h-[5px] rounded-full bg-neutral-400"
              style={{ animation: "live-dot 3.6s cubic-bezier(0.22,0.61,0.36,1) infinite" }}
            />
            <span>{t("Local account auth")}</span>
            <span>·</span>
            <span>KONI-Forge v1.0</span>
          </div>
        </div>
      </div>
    </div>
  );
}
