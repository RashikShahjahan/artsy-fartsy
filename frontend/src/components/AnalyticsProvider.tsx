import React, { createContext, useContext, useEffect } from 'react';
import axios from 'axios';
import { useLocation } from 'react-router-dom';

const ANALYTICS_ENDPOINT = 'https://analytics.rashik.sh/api';

interface EventBase {
  service: string;
  event: string;
  path: string;
  referrer: string;
  user_browser: string;
  user_device: string;
}

interface AnalyticsContextType {
  trackEvent: (eventType: string, properties?: Partial<EventBase>) => void;
}

const AnalyticsContext = createContext<AnalyticsContextType>({
  trackEvent: () => {},
});

export const useAnalytics = (): AnalyticsContextType => useContext(AnalyticsContext);

export const AnalyticsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();
  
  const trackEvent = (eventType: string, properties: Partial<EventBase> = {}) => {
    const event: EventBase = {
      service: 'artsy',
      event: eventType,
      path: location.pathname,
      referrer: document.referrer,
      user_browser: navigator.userAgent,
      user_device: detectDevice(),
      ...properties
    };

    axios.post(ANALYTICS_ENDPOINT, event).catch(error => {
      console.error('Failed to send analytics event', error);
    });
  };

  // Simple device detection
  const detectDevice = (): string => {
    const ua = navigator.userAgent;
    if (/(tablet|ipad|playbook|silk)|(android(?!.*mobi))/i.test(ua)) {
      return 'tablet';
    }
    if (/Mobile|Android|iP(hone|od)|IEMobile|BlackBerry|Kindle|Silk-Accelerated|(hpw|web)OS|Opera M(obi|ini)/.test(ua)) {
      return 'mobile';
    }
    return 'desktop';
  };

  useEffect(() => {
    trackEvent('page_view');
  }, [location.pathname]);

  return (
    <AnalyticsContext.Provider value={{ trackEvent }}>
      {children}
    </AnalyticsContext.Provider>
  );
}; 