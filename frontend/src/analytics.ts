type AnalyticsEventData = Record<string, string | number | boolean>;

declare global {
  interface Window {
    umami?: {
      track: (name: string, data?: AnalyticsEventData) => void;
    };
  }
}

const trackEvent = (name: string, data?: AnalyticsEventData) => {
  if (typeof window !== 'undefined') {
    window.umami?.track(name, data);
  }
};

const analytics = { trackEvent };

export const useAnalytics = () => analytics;
