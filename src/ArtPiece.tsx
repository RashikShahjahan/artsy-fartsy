/*Saving Drawings and viewing gallery(Add auth)
  - On next or previous button click, pass back current artId to backend
  - Backend will return the next or previous artId
  - Update the artId state
  - Pass artid to ArtPiece component
  - ArtPiece component will fetch the drawcommands from backend using artid along with metadata such as username, likes, etc.
  - Artpiece component will render  the drawcommands on a canvas
  Post request to like/unlike drawings
*/

import { Canvas } from "@react-three/fiber";
import { useEffect, useState } from "react";
import { Command } from "./types";
import Line from "./Line";
import Arc from "./Arc";
import { fetchDrawCommands } from "./api";
import { useAuth } from "@clerk/clerk-react";

export default function ArtPiece({artId}: {artId: number}) {
    const [drawCommands, setDrawCommands] = useState<Command[]>([]);
    const { getToken } = useAuth(); //Probably should be passing as ref

    useEffect(() => {
        const fetchCommands = async () => {
            const token = await getToken();
            if (!token) {
                throw new Error('No token found');
            }
            const commands = await fetchDrawCommands(artId, token);
            setDrawCommands(commands);
        };

        fetchCommands();
    }, [artId]);
    return (
    <Canvas>
        {drawCommands.map((command, index) => (
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