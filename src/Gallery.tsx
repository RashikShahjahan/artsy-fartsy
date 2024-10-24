import { useEffect, useState } from "react";
import ArtPiece from "./ArtPiece";
import { ArtData } from "./types";
import { useAuth } from "@clerk/clerk-react";
import { fetchArtCountFromServer, fetchArtFromServer } from "./api";

export default function Gallery() {
    const [artIdx, setArtIdx] = useState(0);
    const [artData, setArtData] = useState<ArtData | null>(null);

    const { getToken } = useAuth();

    useEffect(() => {
        const fetchArt = async () => {
            try {
                const token = await getToken();
                if (!token) {
                    throw new Error('No token found');
                }
                const response = await fetchArtFromServer(artIdx, token);
                setArtData(response.artData[0]);
            } catch (error) {
                console.error("Error fetching art:", error);

            } 
        };

        fetchArt();
        console.log(artData);

    }, [artIdx,artData, getToken]);

    const getPreviousArtId = async () => {
        const token = await getToken();
        if (!token) {
            throw new Error('No token found');
        }
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

    return (
        <div className="w-full h-[calc(100vh-16rem)] bg-white rounded-lg shadow-md p-6 flex flex-col">
            {artData && artData.commands ? (
                <ArtPiece artData={artData} />
            ) : (
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
