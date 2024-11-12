
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
        <button 
          onClick={onToggleDocs}
        >
          Show Docs
        </button>
      </div>
      
      <textarea 
        value={code} 
        onChange={(e) => onCodeChange(e.target.value)} 
        placeholder="# Your code will appear here"
        className="textarea textarea-bordered h-full font-mono"
        disabled={isRunning}
      />
      <div className="card-actions justify-end mt-6">
        <button 
          onClick={onRun}
          className="btn btn-primary w-full"
          disabled={isRunning}
        >
          Run Code
        </button>
      </div>
    </div>
  </div>
); 