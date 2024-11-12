import { Button } from '../common/Button/Button';

interface DrawingCanvasProps {
  image: string;
  isRunning: boolean;
  isSaving: boolean;
  onSave: () => void;
}

export const DrawingCanvas = ({ image, isRunning, isSaving, onSave }: DrawingCanvasProps) => (
  <div className="card w-full md:w-1/2 h-[calc(100vh-20rem)] bg-base-100 shadow-xl">
    <div className="card-body relative">
      {!image && !isRunning && (
        <div className="absolute inset-0 flex items-center justify-center text-center text-gray-500">
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
        className="w-full h-full object-contain mb-16" 
      />
      <Button 
        variant="warning"
        onClick={onSave}
        isLoading={isSaving}
        loadingText="Saving..."
        fullWidth
        className="absolute bottom-6 left-6 right-6"
      >
        Save
      </Button>
    </div>
  </div>
); 