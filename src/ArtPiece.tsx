

import { Canvas } from "@react-three/fiber";
import { ArtData } from "./types";
import Line from "./Line";
import Arc from "./Arc";

export default function ArtPiece({artData}: {artData: ArtData}) {

    return (
    <Canvas>
        {artData.drawCommands.map((command, index) => (
          command.type === 'line' ? 
            <Line key={index} start={[Number(command.args[0]), Number(command.args[1]), 0]} end={[Number(command.args[2]), Number(command.args[3]), 0]} color={command.args[4].toString()} /> :
            <Arc 
              key={index} 
              center={[Number(command.args[0]), Number(command.args[1]), 0]}
              start={[Number(command.args[2]), Number(command.args[3]), 0]}
              end={[Number(command.args[4]), Number(command.args[5]), 0]}
              startAngle={Number(command.args[6])}
              endAngle={Number(command.args[7])}
              clockwise={Boolean(command.args[8])}
              rotation={Number(command.args[9])}
              color={command.args[10].toString()}
            />
        ))}
      </Canvas>
    )
  }