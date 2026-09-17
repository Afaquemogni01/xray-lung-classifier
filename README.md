## EDA findings
- Dataset contains 6432 valid chest x-ray images across 'covid19' 'Normal' 'penumonia'
- The classes are imbalanced , the covid19 cases are in majority while penumonia is minority
- Images have varying sizes and modes, including RGB, grayscale, RGBA, and palette images.
- Some images contain `L`/`R` markers and additional embedded text or details.
- Embedded annotations may cause privacy concerns or allow the model to learn shortcuts rather than lung-related patterns.
- This project is an educational prototype and must not be used for clinical diagnosis.