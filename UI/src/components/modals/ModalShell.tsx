import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "../ui/utils";
import { useT } from "../../i18n";

interface ModalShellProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  subtitle?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
  /** Tailwind max-width for the panel (default max-w-3xl). */
  maxWidthClass?: string;
}

export function ModalShell({
  open,
  onOpenChange,
  title,
  subtitle,
  footer,
  children,
  maxWidthClass = "max-w-3xl",
}: ModalShellProps) {
  const t = useT();
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay
          className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=open]:fade-in-0 data-[state=closed]:fade-out-0"
        />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className={cn(
            "fixed left-1/2 top-1/2 z-50 flex max-h-[85vh] w-[95vw] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-2xl bg-white shadow-2xl ring-1 ring-neutral-200/60 focus:outline-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=open]:fade-in-0 data-[state=closed]:fade-out-0 data-[state=open]:zoom-in-95 data-[state=closed]:zoom-out-95 duration-200",
            maxWidthClass,
          )}
        >
          {/* Header — 3px accent bar, title, close */}
          <div className="relative flex h-14 shrink-0 items-center justify-between border-b border-neutral-100 bg-neutral-50 pl-5 pr-3">
            <span className="absolute left-0 top-0 h-full w-[3px] bg-neutral-900" aria-hidden />
            <div className="min-w-0">
              <DialogPrimitive.Title className="truncate text-[17px] font-semibold tracking-tight text-neutral-900">
                {title}
              </DialogPrimitive.Title>
              {subtitle && <div className="truncate text-[12px] text-neutral-400">{subtitle}</div>}
            </div>
            <DialogPrimitive.Close
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-neutral-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
            >
              <X className="h-4 w-4" />
              <span className="sr-only">{t("Close")}</span>
            </DialogPrimitive.Close>
          </div>

          {/* Body — scrollable */}
          <div className="flex-1 overflow-y-auto">{children}</div>

          {/* Footer — optional, right-aligned */}
          {footer && (
            <div className="flex h-16 shrink-0 items-center justify-end gap-2 border-t border-neutral-100 bg-neutral-50 px-6">
              {footer}
            </div>
          )}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
