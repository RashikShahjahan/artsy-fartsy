import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import Line from "./Line";
import Arc from "./Arc";
import { ArtData } from "./types";
import { useAuth } from "@clerk/clerk-react";
import { fetchArtCountFromServer, fetchArtFromServer, isLikedRequest, toggleLikeRequest } from "./api";

export default function Gallery() {
    const [artIdx, setArtIdx] = useState(0);
    const [artData, setArtData] = useState<ArtData | null>(null);
    const [isLiked, setIsLiked] = useState(false);

    const { getToken } = useAuth();

    const getPreviousArtId = async () => {
        if (artIdx > 0) {
            setArtIdx(artIdx - 1);

        }
        
    }

    const getNextArtId = async () => {
        const token = await getToken();
        if (!token) {
            throw new Error('No token found');
        }

        const data = await fetchArtCountFromServer(token);
        console.log(data);
        if (artIdx < data.artCount-1) {
            setArtIdx(artIdx + 1);
        }

    }

    const toggleLike = async () => {
        const token = await getToken();
        if (!token) {
            throw new Error('No token found');
        }

        if (artData) {
            await toggleLikeRequest(artData.id, token);
            const response = await isLikedRequest(artData.id, token);
            setIsLiked(response.isLiked);
        } else {
            console.error('No art data available');
        }
    }

    useEffect(() => {
        const fetchArt = async () => {
            const token = await getToken();
            if (!token) {
                throw new Error('No token found');
            }
            const response = await fetchArtFromServer(artIdx, token);
            const data = response.artData[0];
            setArtData(data);
        };

        fetchArt();

    }, [artIdx,isLiked,getToken]);

    useEffect(() => {
        const checkIsLiked = async () => {
            if (artData) {
                const token = await getToken();
                if (!token) {
                    throw new Error('No token found');
                }
                const response = await isLikedRequest(artData.id, token);
                setIsLiked(response.isLiked);
            }
        };

        checkIsLiked();
    }, [artData, getToken]);


    
    return (
        <div className="w-full h-[calc(100vh-16rem)] bg-white rounded-lg shadow-md p-6 flex flex-col">
            {artData && artData.commands ? (
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
                                onClick={() => toggleLike()} 
                                className={`px-4 py-2 rounded-full transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-pink-200 focus:ring-opacity-50 ${
                                    isLiked 
                                        ? 'bg-pink-500 text-white hover:bg-pink-600' 
                                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                }`}
                            >
                                {isLiked ? 'Liked' : 'Like'}
                            </button>
                            <span className="ml-2 text-lg font-semibold">{artData.likes}</span>
                        </div>
                    </div>
                    </div>            ) : (
                <div>No art data available</div>
            )}
            <div className="flex flex-row justify-between">
                <button className="w-1/2 bottom-6 left-6 right-6 px-6 py-3 bg-yellow-400 text-black rounded-lg hover:bg-yellow-300 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-yellow-200 focus:ring-opacity-50"
                    onClick={() => getPreviousArtId()}
                >
                    Previous
                </button>
                <button 
                    className="w-1/2 bottom-6 left-6 right-6 px-6 py-3 bg-pink-400 text-black rounded-lg hover:bg-pink-300 transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-pink-200 focus:ring-opacity-50"
                    onClick={() => getNextArtId()}
                >
                    Next
                </button>
            </div>
        </div>
    )
}
