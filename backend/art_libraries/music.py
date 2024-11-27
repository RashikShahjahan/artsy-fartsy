import musicpy as mp

class Chord:
    """A class representing a musical chord using musicpy.
    
    Attributes:
        chord: A musicpy chord object
    """
    
    def __init__(self, root_note: str, chord_type: str):
        """Initialize a new chord.
        
        Args:
            root_note (str): The root note of the chord (e.g., 'C', 'D6')
            chord_type (str): The type of chord (e.g., 'm7', 'maj7')
        """
        self.chord = mp.get_chord(start=root_note, current_chord_type=chord_type)

    def raise_pitch(self, semitones: int) -> None:
        """Raise the pitch of the chord by a specified number of semitones.
        
        Args:
            semitones (int): Number of semitones to raise the pitch
        """
        self.chord.up(semitones)

    def lower_pitch(self, semitones: int) -> None:
        """Lower the pitch of the chord by a specified number of semitones.

        Args:
            semitones (int): Number of semitones to lower the pitch
        """
        self.chord.down(semitones)

    def set_duration(self, duration: float | list[float]) -> None:
        """Set the duration of the chord notes.
        
        Args:
            duration (float | list[float]): Duration value(s) in beats. Can be a single float
                for uniform duration or a list of floats for different durations per note.
        """
        self.chord.set(duration=duration)

    def set_interval(self, interval: float | list[float]) -> None:
        """Set the interval between chord notes.
        
        Args:
            interval (float | list[float]): Interval value(s) in beats. Can be a single float
                for uniform intervals or a list of floats for different intervals between notes.
        """
        self.chord.set(interval=interval)


class ChordProgression:
    """A class representing a sequence of musical chords.
    
    Attributes:
        chords (list[Chord]): List of Chord objects in the progression
    """
    
    def __init__(self):
        """Initialize an empty chord progression."""
        self.chords = []
    
    def add_chord(self, root_note: str, chord_type: str) -> None:
        """Add a new chord to the progression.
        
        Args:
            root_note (str): The root note of the chord (e.g., 'C', 'D6')
            chord_type (str): The type of chord (e.g., 'm7', 'maj7')
        """
        chord = Chord(root_note, chord_type)
        self.chords.append(chord)
    
    def save(self, filename: str = "output.mid") -> None:
        """Save the chord progression as a MIDI file.
        
        Args:
            filename (str): Path where the MIDI file will be saved.
                Defaults to "output/progression.mid"
                
        Raises:
            ValueError: If the chord progression is empty
        """
        if not self.chords:
            raise ValueError("Cannot save empty progression")
        progression = mp.concat([chord.chord for chord in self.chords])
        mp.write(progression, name=filename)



