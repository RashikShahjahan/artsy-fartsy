import { Outlet } from 'react-router-dom';

function App() {
  return (
    <div className="flex flex-col gap-6 container mx-auto p-6">
      <Outlet />
    </div>
  );
}

export default App;
