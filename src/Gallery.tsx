

import { useEffect, useState } from "react";
import ArtPiece from "./ArtPiece";
import { ArtData} from "./types";
import { useAuth } from "@clerk/clerk-react";
import { fetchArtFromServer, fetchNextArtId, fetchPreviousArtId } from "./api";

export default function Gallery() {
    const [artId, setArtId] = useState(0);
    const [artData, setArtData] = useState<ArtData>({
        drawCommands: [
            { type: 'line', args: [0, 0, 2, 0, 'black'] },
            { type: 'line', args: [2, 0, 2, 2, 'black'] },
            { type: 'line', args: [2, 2, 0, 2, 'black'] },
            { type: 'line', args: [0, 2, 0, 0, 'black'] }
        ],
        username: "",
        likes: 0
    });

    const { getToken } = useAuth();
    useEffect(() => {
        const fetchArt = async () => {
            const token = await getToken();
            if (!token) {
                throw new Error('No token found');
            }
            const response = await fetchArtFromServer(artId, token);
            setArtData(response.artData);
            setArtId(response.artId);
        };

        fetchArt();
    }, [artId]);

    const getPreviousArtId = async () => {
        const token = await getToken();
        if (!token) {
            throw new Error('No token found');
        }
        const response = await fetchPreviousArtId(token);
        setArtId(response.artId);
    }

    const getNextArtId = async () => {
        const token = await getToken();
        if (!token) {
            throw new Error('No token found');
        }
        const response = await fetchNextArtId(token);
        setArtId(response.artId);
    }

    return (
        <div className="w-full h-[calc(100vh-16rem)] bg-white rounded-lg shadow-md p-6 flex flex-col">
            <ArtPiece artData={artData} />
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