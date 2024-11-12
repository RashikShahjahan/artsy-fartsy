import { Button } from '../common/Button/Button';

interface CodeEditorProps {
  code: string;
  onCodeChange: (code: string) => void;
  isRunning: boolean;
  onRun: () => void;
  onToggleDocs: () => void;
}

export const CodeEditor = ({ 
  code, 
  onCodeChange, 
  isRunning, 
  onRun, 
  onToggleDocs 
}: CodeEditorProps) => (
  <div className="card w-full md:w-1/2 h-[calc(100vh-20rem)] bg-base-100 shadow-xl">
    <div className="card-body">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-semibold">Generated Code</h3>
        <Button 
          variant="ghost"
          size="sm"
          onClick={onToggleDocs}
        >
          Show Docs
        </Button>
      </div>
      
      <textarea 
        value={code} 
        onChange={(e) => onCodeChange(e.target.value)} 
        placeholder="# Your code will appear here"
        className="textarea textarea-bordered h-full font-mono"
        disabled={isRunning}
      />
      <div className="card-actions justify-end mt-6">
        <Button 
          onClick={onRun}
          isLoading={isRunning}
          loadingText="Running..."
          fullWidth
        >
          Run Code
        </Button>
      </div>
    </div>
  </div>
); 