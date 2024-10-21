import { Canvas } from '@react-three/fiber'
import Line from './Line';
/*Tasks:
Setup react-three/fiber with react 
    -Create draw function for ARC and LINE
    -Call draw function from react
    -Add textbox in react to call interpreter on button click
    -Create render function to render the output list of objects

Write version of interpreter to run single command->Line and arc:
    -Create lexer for ARC and LINE
    -Parse tokens to create ARC and LINE objects
    -return ARC and LINE objects

Enhancements:
    Add variables and expressions
    Add loops
    Add error validation
    Add coloring
*/
function App() {
  return (
    <Canvas>
        <Line start={[0, 0, 0]} end={[1, 1, 1]} />
    </Canvas>
  )
}

export default App;