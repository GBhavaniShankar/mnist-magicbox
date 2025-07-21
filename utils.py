from typing import List, Tuple
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self, padding: float = 0.2):
        """
        :param padding: fraction of width/height to pad around each component before centering
        """
        self.directions = [(-1,0), (1,0), (0,-1), (0,1)]
        self.padding = padding

    def dfs(self,
            image: np.ndarray,
            visited: np.ndarray,
            start: Tuple[int,int],
            component: List[Tuple[int,int]]):
        """Depth‐first search collecting connected pixels with value > 0."""
        rows, cols = image.shape
        stack = [start]
        while stack:
            x, y = stack.pop()
            if (0 <= x < rows and 0 <= y < cols
               and not visited[x,y]
               and image[x,y] > 0):
                visited[x,y] = True
                component.append((x, y))
                for dx, dy in self.directions:
                    stack.append((x + dx, y + dy))

    def find_components(self, image: np.ndarray) -> List[List[Tuple[int,int]]]:
        """Return list of connected‐component pixel lists from grayscale array."""
        rows, cols = image.shape
        visited = np.zeros((rows, cols), dtype=bool)
        components: List[List[Tuple[int,int]]] = []
        for i in range(rows):
            for j in range(cols):
                if image[i,j] > 0 and not visited[i,j]:
                    comp: List[Tuple[int,int]] = []
                    self.dfs(image, visited, (i,j), comp)
                    components.append(comp)
        return components

    def make_28x28(self,
                   image: np.ndarray,
                   pixels: List[Tuple[int,int]]) -> np.ndarray:
        """
        Crop to component bounding box, pad, center on square canvas, resize to 28×28.
        Grayscale values are preserved.
        """
        rs, cs = zip(*pixels)
        minr, maxr = min(rs), max(rs)
        minc, maxc = min(cs), max(cs)
        h, w = maxr - minr + 1, maxc - minc + 1

        # Crop
        crop = image[minr:maxr+1, minc:maxc+1]

        # Pad by fraction
        ph, pw = int(h * self.padding), int(w * self.padding)
        padded = np.zeros((h + 2*ph, w + 2*pw), dtype=np.uint8)
        padded[ph:ph+h, pw:pw+w] = crop

        # Center on square canvas
        side = max(padded.shape)
        square = np.zeros((side, side), dtype=np.uint8)
        yoff = (side - padded.shape[0]) // 2
        xoff = (side - padded.shape[1]) // 2
        square[yoff:yoff+padded.shape[0], xoff:xoff+padded.shape[1]] = padded

        # Resize
        pil = Image.fromarray(square)
        resized = pil.resize((28, 28), Image.Resampling.LANCZOS)
        return np.array(resized)

    def process_image(self, pil_img: Image.Image) -> List[np.ndarray]:
        """
        :param pil_img: any-mode PIL image (will be converted to ’L’)
        :return: list of 28×28 numpy arrays, one per connected component
        """
        gray = np.array(pil_img.convert('L'))
        components = self.find_components(gray)
        crops = [self.make_28x28(gray, comp) for comp in components]
        return crops
