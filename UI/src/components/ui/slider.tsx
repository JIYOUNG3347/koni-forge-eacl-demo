"use client"

import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"

import { cn } from "./utils"

function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  color = "neutral", // Default to neutral for non-home pages
  ...props
}: React.ComponentProps<typeof SliderPrimitive.Root> & {
  color?: "primary" | "accent" | "chart-5" | "dataset" | "neutral"
}) {
  const _values = React.useMemo(
    () =>
      Array.isArray(value)
        ? value
        : Array.isArray(defaultValue)
        ? defaultValue
        : [min, max],
    [value, defaultValue, min, max],
  )

  const colorClasses = {
    neutral: {
      range: "bg-gray-600",
      thumb: "border-gray-600 ring-gray-600/50",
    },
    primary: {
      range: "bg-primary",
      thumb: "border-primary ring-primary/50",
    },
    accent: {
      range: "bg-accent",
      thumb: "border-accent ring-accent/50",
    },
    "chart-5": {
      range: "bg-chart-5",
      thumb: "border-chart-5 ring-chart-5/50",
    },
    dataset: {
      range: "bg-dataset",
      thumb: "border-dataset ring-dataset/50",
    },
  }

  return (
    <SliderPrimitive.Root
      data-slot="slider"
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      className={cn(
        "relative flex w-full touch-none items-center select-none data-[disabled]:opacity-50 data-[orientation=vertical]:h-full data-[orientation=vertical]:min-h-44 data-[orientation=vertical]:w-auto data-[orientation=vertical]:flex-col",
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track
        data-slot="slider-track"
        className={cn(
          "bg-muted relative grow overflow-hidden rounded-full data-[orientation=horizontal]:h-4 data-[orientation=horizontal]:w-full data-[orientation=vertical]:h-full data-[orientation=vertical]:w-1.5",
        )}
      >
        <SliderPrimitive.Range
          data-slot="slider-range"
          className={cn(
            "absolute data-[orientation=horizontal]:h-full data-[orientation=vertical]:w-full",
            colorClasses[color]?.range
          )}
        />
      </SliderPrimitive.Track>
      {Array.from({ length: _values.length }, (_, index) => (
        <SliderPrimitive.Thumb
          data-slot="slider-thumb"
          key={index}
          className={cn(
            "bg-background block size-4 shrink-0 rounded-full border shadow-sm transition-[color,box-shadow] hover:ring-4 focus-visible:ring-4 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50",
            colorClasses[color]?.thumb
          )}
        />
      ))}
    </SliderPrimitive.Root>
  )
}

export { Slider }