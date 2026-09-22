import { forwardRef } from "react";
import type { LucideProps } from "lucide-react";

const KoniForgeIcon = forwardRef<SVGSVGElement, LucideProps>(
  (
    {
      color = "currentColor",
      size = 24,
      strokeWidth = 1.5,
      absoluteStrokeWidth,
      className,
      ...rest
    },
    ref
  ) => {
    const computedStrokeWidth =
      absoluteStrokeWidth
        ? Number(strokeWidth)
        : (Number(strokeWidth) * 24) / Number(size);

    return (
      <svg
        ref={ref}
        xmlns="http://www.w3.org/2000/svg"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke={color}
        strokeWidth={computedStrokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
        {...rest}
      >
        {/* Outer circle */}
        <circle cx="12" cy="12" r="10.5" strokeOpacity={0.2} />

        {/* Hexagon */}
        <polygon
          points="22,12 17,3.8 7,3.8 2,12 7,20.2 17,20.2"
          strokeOpacity={0.6}
        />

        {/* Three diagonals through the hexagon centre */}
        <line x1="22" y1="12" x2="2" y2="12" strokeOpacity={0.2} />
        <line x1="17" y1="3.8" x2="7" y2="20.2" strokeOpacity={0.2} />
        <line x1="7" y1="3.8" x2="17" y2="20.2" strokeOpacity={0.2} />

        {/* Inner circle */}
        <circle cx="12" cy="12" r="6" strokeOpacity={0.4} />

        {/* K letterform, drawn as strokes */}
        {/* Vertical bar */}
        <line
          x1="9.5"
          y1="8.2"
          x2="9.5"
          y2="15.8"
          stroke={color}
          strokeWidth={computedStrokeWidth * 1.4}
          strokeOpacity={1}
        />
        {/* Upper diagonal */}
        <line
          x1="9.5"
          y1="12"
          x2="14.5"
          y2="8.2"
          stroke={color}
          strokeWidth={computedStrokeWidth * 1.4}
          strokeOpacity={1}
        />
        {/* Lower diagonal */}
        <line
          x1="9.5"
          y1="12"
          x2="14.5"
          y2="15.8"
          stroke={color}
          strokeWidth={computedStrokeWidth * 1.4}
          strokeOpacity={1}
        />
      </svg>
    );
  }
);

KoniForgeIcon.displayName = "KoniForgeIcon";

export { KoniForgeIcon };
