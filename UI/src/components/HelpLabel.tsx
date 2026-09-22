/**
 * HelpLabel — Inline ? icon with tooltip for explaining terms.
 *
 * Props: text (tooltip content), children (label content)
 * Uses Tooltip from ui/tooltip.
 */

import React from "react";
import { HelpCircle } from "lucide-react";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "./ui/tooltip";

interface HelpLabelProps {
  text: string;
  children: React.ReactNode;
}

export function HelpLabel({ text, children }: HelpLabelProps) {
  return (
    <span className="inline-flex items-center gap-1">
      {children}
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            className="inline-flex items-center justify-center text-neutral-400 hover:text-neutral-600 transition-colors"
            tabIndex={-1}
          >
            <HelpCircle className="w-3.5 h-3.5" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-[280px] text-xs leading-relaxed">
          {text}
        </TooltipContent>
      </Tooltip>
    </span>
  );
}
