import { useEffect } from 'react';
import { useAnalytics } from 'rashik-analytics-provider';

interface SimilarDrawingsProps {
  drawings: string[];
  isFinding: boolean;
}

export const SimilarDrawings = ({ drawings, isFinding }: SimilarDrawingsProps) => {
  const { trackEvent } = useAnalytics();

  useEffect(() => {
    if (drawings.length > 0) {
      trackEvent('search_results_displayed', { results_count: drawings.length });
    }
  }, [drawings, trackEvent]);

  const handleImageClick = (index: number) => {
    trackEvent('search_result_clicked', { result_index: index });
  };

  return (
    <div className="w-full grid grid-cols-1 md:grid-cols-3 gap-6">
      {drawings.length === 0 && !isFinding && (
        <div className="col-span-full text-center text-gray-500 py-12">
          <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <p>Enter a description to find similar drawings</p>
        </div>
      )}
      {drawings.map((drawing, index) => (
        <div 
          key={index} 
          className="card w-full bg-base-100 shadow-xl cursor-pointer"
          onClick={() => handleImageClick(index)}
        >
          <div className="card-body p-4">
            <img 
              src={drawing}
              alt={`Similar Image ${index}`}
              className="w-full h-auto object-contain" 
            />
          </div>
        </div>
      ))}
    </div>
  );
} 