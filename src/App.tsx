import { Canvas } from '@react-three/fiber'
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
        <mesh>
          <boxGeometry args={[2, 2, 2]} />
          <meshStandardMaterial />
        </mesh>
        <ambientLight intensity={0.1} />
        <directionalLight position={[0, 0, 5]} color="red" />
      </Canvas>
  )
}

export default App;