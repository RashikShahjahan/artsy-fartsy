import { SignedIn, SignedOut, SignIn,  UserButton } from "@clerk/clerk-react";
import DrawingBoard from "./DrawingBoard";


export default function App() {

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-100 p-4 sm:p-6 md:p-8">
      <div className="w-full max-w-7xl">
        <div className="fixed top-4 right-4 sm:top-6 sm:right-6 md:top-8 md:right-8 z-10">
          <UserButton />
        </div>
        <SignedIn>
          <DrawingBoard/>
        </SignedIn>
        <SignedOut>
            <SignIn />
        </SignedOut>
      </div>
    </div>
  )
}
