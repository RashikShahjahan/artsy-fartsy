import { Canvas } from "@react-three/fiber";
import { ArtData } from "./types";
import Line from "./Line";
import Arc from "./Arc";
import { useEffect, useState } from "react";
import { fetchLikedStatus, fetchLikes, likeArtRequest, unlikeArtRequest } from "./api";
import { useAuth } from "@clerk/clerk-react";

export default function ArtPiece({artData}: {artData: ArtData}) {
    const [isLiked, setIsLiked] = useState(false);
    const [likes, setLikes] = useState(artData.likes);
    const { getToken } = useAuth();

    useEffect(() => {
        const checkIfLiked = async () => {
            const token = await getToken();
            if (!token) {
                return;
            }
            const response = await fetchLikedStatus(artData.id, token);
            setIsLiked(response.isLiked);
            const likesResponse = await fetchLikes(artData.id, token);
            setLikes(likesResponse.likes);
        };
        checkIfLiked();
    }, []);

    const likeArt = async () => {
        const token = await getToken();
        if (!token) {
            throw new Error('No token found');
        }
        setIsLiked(true);
        setLikes(likes + 1);
        await likeArtRequest(artData.id,token);
    }

    const unlikeArt = async () => {
        const token = await getToken();
        if (!token) {
            throw new Error('No token found');
        }
        setIsLiked(false);
        setLikes(likes - 1);
        await unlikeArtRequest(artData.id,token);
    }

    return (
        <div className="flex flex-col h-full">
            <Canvas className="flex-grow">
                {artData.commands.map((command, index) => (
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
                            color={command.args[10].toString()} />
                ))}
            </Canvas>
            <div className="mt-4 p-2 flex items-center justify-end">
                <div className="flex items-center">
                    <button 
                        onClick={() => isLiked ? unlikeArt() : likeArt()} 
                        className={`px-4 py-2 rounded-full transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-pink-200 focus:ring-opacity-50 ${
                            isLiked 
                                ? 'bg-pink-500 text-white hover:bg-pink-600' 
                                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                    >
                        {isLiked ? 'Liked' : 'Like'}
                    </button>
                    <span className="ml-2 text-lg font-semibold">{likes}</span>
                </div>
            </div>
        </div>
    )
}
