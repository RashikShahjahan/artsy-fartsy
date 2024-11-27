from music import ChordProgression

def test_create_progression():
    # Create a new chord progression
    progression = ChordProgression()
    
    # Add some chords - this creates a common I-V-vi-IV progression in C major
    progression.add_chord('C', 'maj')  # C major
    progression.add_chord('G', 'maj')  # G major
    progression.add_chord('A', 'm')    # A minor
    progression.add_chord('F', 'maj')  # F major
    
    # Save the progression to a MIDI file
    progression.save("output/test_progression.mid")

if __name__ == "__main__":
    test_create_progression()

