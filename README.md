# Plumcot
Code for Plumcot project which don't fit https://github.com/pyannote nor https://github.com/PaulLerner/gecko

See also https://github.com/PaulLerner/Hard-Alignment which unfortunately contains closed source dependencies.

## Utils

Get the distances between every speech turn embeddings for every speaker using pyannote.core.utils.distance. Allows to sort speech turns of a given speaker depending on their distance from the cluster center.
