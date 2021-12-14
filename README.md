~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
<br> This project aims to become an accurate song labeling program
<br><br>
<ol>
<li> It breaks a song into parts depending on rhythmic patterns.</li>
<li> It breaks each part into meters and beats depending on the time signature of the part.</li>
<li> It infers the dominant frequencies of each beat with 3 approaches:
<ul>
<li>crepe (tensorflow)</li>
<li>probabilistic yin algorithm</li>
<li>manually - extracting spectrogram values and applying a loudness threshold</li>
</ul>
<li> It finds:
<ul>
<li> Correlations between dominant frequencies for each beat understanding chords</li>
<li> Correlations between subsequent beats and meters</li>
<li> Correlations between subsequent beats and meters</li>
</ul>
<li> It extracts the "mood" of the song based on the correlations and the rhythmic patterns.
</ol>
<br> eg. If a song is being dominated by Major chord occurrences it will probably be a happy song.
<br> If the beats per minute are higher than a threshold (eg. 95) it is probably an energetic song.
<br> This project is still under development...
<br>~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
