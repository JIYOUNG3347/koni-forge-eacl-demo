import React from 'react';
import { t } from "../i18n";

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class FaultBoundary extends React.Component<
  { children: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('[ErrorBoundary] Uncaught error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center min-h-screen p-8 text-center">
          <h1 className="text-2xl font-bold text-foreground mb-4">
            {t("Something went wrong")}
          </h1>
          <p className="text-muted-foreground mb-4">
            {t("An unexpected error occurred. Please refresh the page.")}
          </p>
          {this.state.error && (
            <pre className="text-left text-xs text-red-500 bg-red-50 p-4 rounded-lg mb-6 max-w-xl overflow-auto max-h-40">
              {this.state.error.message}
              {"\n"}
              {this.state.error.stack?.split("\n").slice(0, 5).join("\n")}
            </pre>
          )}
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity"
          >
            {t("Refresh")}
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
