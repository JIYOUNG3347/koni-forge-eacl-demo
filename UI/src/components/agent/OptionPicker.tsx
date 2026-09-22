/**
 * OptionPicker — parses `**Option N: NAME**` markdown blocks out of an
 * assistant message and renders each as a clickable chip. Clicking a chip
 * sends the choice back to the dispatcher via `dispatchMessage("Option N")`,
 * which matches boundary specialist's resume regex `^option\s*(\d+)`.
 */

import React from "react";
import { ArrowRight, Star } from "lucide-react";
import { useT } from "../../i18n";

export interface AgentOption {
  number: number;
  name: string;
  description: string;
  isRecommended: boolean;
}

/** Extract `**Option N: NAME**\n<description?>` blocks from a content string. */
export function parseAgentOptions(content: string): AgentOption[] {
  if (!content || !content.includes("**Option")) return [];
  const lines = content.split("\n");
  const out: AgentOption[] = [];

  for (let i = 0; i < lines.length; i++) {
    const m = /^\s*\*\*Option\s+(\d+):\s*(.+?)\*\*\s*(.*?)\s*$/.exec(lines[i]);
    if (!m) continue;
    const number = parseInt(m[1], 10);
    let name = m[2].trim();
    let trailing = m[3].trim(); // anything on the same line after `**`

    // description = the next non-empty line (until another Option or blank)
    let description = trailing;
    for (let j = i + 1; j < lines.length; j++) {
      const next = lines[j].trim();
      if (!next) break;
      if (/^\*\*Option\s+\d+/.test(next)) break;
      description = description ? `${description} ${next}` : next;
      break;
    }

    const isRecommended = /recommend/i.test(description) || /recommend/i.test(trailing);
    out.push({ number, name, description, isRecommended });
  }
  return out;
}

interface OptionPickerProps {
  options: AgentOption[];
  disabled?: boolean;
  onPick: (option: AgentOption) => void;
}

export function OptionPicker({ options, disabled, onPick }: OptionPickerProps) {
  const t = useT();
  if (!options.length) return null;
  return (
    <div className="mt-2 flex flex-col gap-1.5">
      {options.map((opt) => (
        <button
          key={opt.number}
          type="button"
          disabled={disabled}
          onClick={() => onPick(opt)}
          className={`group w-full text-left flex items-center gap-2 px-2.5 py-1.5 rounded-lg border transition-all
            ${opt.isRecommended
              ? "border-blue-300 bg-blue-50 hover:bg-blue-100"
              : "border-neutral-200 bg-white hover:bg-neutral-50"}
            ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
          `}
        >
          <span className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold
            ${opt.isRecommended ? "bg-blue-600 text-white" : "bg-neutral-200 text-neutral-700"}
          `}>
            {opt.number}
          </span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1">
              <p className={`text-[11.5px] font-semibold truncate ${opt.isRecommended ? "text-blue-800" : "text-neutral-800"}`}>
                {opt.name}
              </p>
              {opt.isRecommended && (
                <span className="inline-flex items-center gap-0.5 text-[9px] font-bold text-blue-700 bg-blue-100 px-1 py-px rounded">
                  <Star className="w-2.5 h-2.5" fill="currentColor" strokeWidth={0} /> {t("Recommended")}
                </span>
              )}
            </div>
            {opt.description && (
              <p className="text-[10px] text-neutral-500 truncate">{opt.description}</p>
            )}
          </div>
          <ArrowRight className={`w-3 h-3 flex-shrink-0 transition-transform
            ${opt.isRecommended ? "text-blue-500" : "text-neutral-300 group-hover:text-neutral-500 group-hover:translate-x-0.5"}
          `} />
        </button>
      ))}
    </div>
  );
}
