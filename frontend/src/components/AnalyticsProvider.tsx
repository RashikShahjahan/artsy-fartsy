import React, { createContext, useContext, useEffect } from 'react';
import axios from 'axios';
import { useLocation } from 'react-router-dom';

const ANALYTICS_ENDPOINT = 'https://analytics.rashik.sh/api';

type Properties = Record<string, any>;

interface AnalyticsContextType {
  trackEvent: (eventType: string, properties?: Properties) => void;
}

const AnalyticsContext = createContext<AnalyticsContextType>({
  trackEvent: () => {},
});

export const useAnalytics = (): AnalyticsContextType => useContext(AnalyticsContext);

export const AnalyticsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();
  const userId = localStorage.getItem('user_id') || 'anonymous';

  const trackEvent = (eventType: string, properties: Properties = {}) => {
    axios.post(ANALYTICS_ENDPOINT, {
      event_type: eventType,
      user_id: userId,
      properties,
    }).catch(error => {
      console.error('Failed to send analytics event', error);
    });
  };

  useEffect(() => {
    trackEvent('page_view', { page: location.pathname });
  }, [location.pathname]);

  return (
    <AnalyticsContext.Provider value={{ trackEvent }}>
      {children}
    </AnalyticsContext.Provider>
  );
}; 