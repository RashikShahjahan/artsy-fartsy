import DrawingBoard from "./DrawingBoard";


export default function App() {

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-100 p-4 sm:p-6 md:p-8">
      <div className="w-full max-w-7xl">
          <DrawingBoard/>
      </div>
    </div>
  )
}
