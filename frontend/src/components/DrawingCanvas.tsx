import { useEffect } from 'react';
import { useAnalytics } from 'rashik-analytics-provider';

interface DrawingCanvasProps {
  image: string;
  isRunning: boolean;
  isSaving: boolean;
  onSave: () => void;
}

export const DrawingCanvas = ({ image, isRunning, isSaving, onSave }: DrawingCanvasProps) => {
  const { trackEvent } = useAnalytics();

  useEffect(() => {
    if (image) {
      trackEvent('image_displayed');
    }
  }, [image, trackEvent]);

  return (
    <div className="card w-full md:w-1/2 h-[calc(100vh-20rem)] bg-base-100 shadow-xl">
      <div className="card-body">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-lg font-semibold">Drawing Canvas</h3>
        </div>

        <div className="h-[calc(100%-4rem)] mb-4 pt-4">
          {!image && !isRunning && (
            <div className="absolute inset-4 flex items-center justify-center text-center text-gray-500">
              <div>
                <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p>Your drawing will appear here</p>
              </div>
            </div>
          )}
          <img 
            src={image}
            className="w-full h-full object-contain border rounded-lg border-base-300" 
          />
        </div>

        <div className="card-actions justify-end">
          <button 
            onClick={onSave}
            className="btn btn-warning w-full"
            disabled={isSaving}
          >
            {isSaving ? (
              <>
                <span className="loading loading-spinner"></span>
                Saving...
              </>
            ) : "Save to public gallery"}
          </button>
        </div>
      </div>
    </div>
  );
} 