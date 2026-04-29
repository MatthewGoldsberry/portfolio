# The Fellowship of the Script: An interactive visualization of LotR dialogues

**Team Members:** [Matthew Goldsberry](https://github.com/MatthewGoldsberry) & [Isaac Dowdy](https://github.com/isaac-dowdy)

**Links:** [Live Application](https://the-fellowship-of-the-script.vercel.app/) | [Source Code (GitHub)](https://github.com/MatthewGoldsberry/Movie-Time) | [Demo Video](#video-demonstration)

---

## Project Overview & Motivations

...

* **The Problem:** ...
* **The Goal:** ...

## Video Demonstration

<figure markdown="span">
  <video controls loop muted playsinline width="700">
    <source src="" type="video/mp4"> 
    Your browser does not support the video tag.
  </video>
</figure>

## The Data

...

## Design Process & Early Sketches

...

## Visual Components & Interactions

...

## Key Discoveries & Findings

...

## Technical Implementation

### Tech Stack

#### Python (Data Exploration)

...

#### JavaScript, HTML, CSS (Frontend)

...

### Architecture

#### `data/`

...

#### `js/` (JavaScript)

...

#### `css/` (CSS)

Contains the style sheet files for styling our application.

### Running Locally

To run this locally:

1. Clone the repository and navigate to the directory

    ```bash
    git clone https://github.com/MatthewGoldsberry/Movie-Time.git
    cd Movie-Time
    ```

2. Launch a local web server (We used Python to do this)

    ```bash
    python -m http.server
    ```

3. Access the application at [`http://localhost:8000`](http://localhost:8000)

## Challenges & Future Work

### Challenges

There were a couple of distinct challenges encountered in this project. And they all can be generalized into two categories, data interpretation and effective visualizations. These often went hand in hand with trying to figure out how to interpret AND represent the data in the most effective manner.

When it came to data interpretation, one of the hardest changes was trying to programmatically determine top phrases of each character. How do you even effectively label something as a phrase and not just a random collection of words? Then how do you determine from that which of those "phrases" are unique / identifiable with that character? For this problem we decided that a phrase could be any grouping of consecutive words within a sentence of length 2-8. From these we than tried to mathematically determine some level of uniqueness score to it between all of the characters and then grabbed the most visible frequently occurring ones. Then to make it to the word cloud representation of these, there had to be more than 3 occurrences of that phrase. This approach was aided by AI and some reading, but ultimately got capped at some point due to complexity and time constraints. This provides some insightful findings but doesn't one hundred percent accomplish what a standard expectation would be for the question "What are common phrases of the characters". This is a big thing that we want to look at in future work.

Site coloring also proved to be a tricky challenge throughout this project too. We wanted a theme that matched the ancient map idea which mean a lot of shades of brown and more toned down colors. This becomes a major problem with color theory because of need to distinctiveness and visibility. For this project there was a lot of trying to strike the right balance in tradeoffs between keeping it visually pleasing and aligned with the goal them, while trying to best follow the guidelines of color theory. This also proved difficult from the sense of needing a lot more CSS elements because there needed to be layers to effectively visualize some of the things and features distinctively which a more monotone color setup.

### Future Work

* **Improved Top Phrase Identification:** This was mentioned in our [challenges](#challenges) section but this is something that would be very insightful and beneficial to get in a better state from both a user perspective, and for us to be able to learn those concepts surrounding NLP.
* **Improved Path Representation:** The LotR is all over the place in terms of character locations, they travel a lot. A bi-product of this and the time constraints meant the map representations could not be 100% down to the exact location or path taken. Specifically making the paths more exact would be an extremely cool feature to have down to be able to know the exact routes of the fellowship.
* **Improved Character Representation on Map:** We currently only show the characters that are in the scene on the map. Another great feature would be carefully orchestrating all character nodes on the map at once, regardless of if they are in the scene or not, and passively update their location if they are moving in the background.
* **Data-Specific Timeline:** The scene slider is super beneficial for high level looks at the text analysis, but it would be another layer of information if we could change the slider to be days and be able to provide a better temporal understanding of the characters location and journey over time. These could then be binned by the scene that they are in for determination of the higher-level textual analysis of the scenes and characters.
* **Addition of More Characters:** There are a lot of characters that were not represented in this visualization. Expanding beyond just the fellowship would be beneficial for further text analysis and understanding, and would be awesome when paired with the improved character orchestration on the maps and being able to properly play out the scenes.
* **Scene-Level View:** To make the scene player more accurate in depicting who is talking to who, it would be beneficial to support a zoomed in view where character locations could more accurately depicting so natural groupings within scenes can be seen.
* **Improved Representation Accuracy:** Right now a lot of stats, such as scenes present are based on the character saying a line in that scene, this is not always the case in movies. It is possible, and likely that there are characters in scenes that do not have any lines for that specific scene.

## Acknowledgements & AI Usage

We would like to extend our appreciation to Dr. Aurisano for providing valuable feedback and guidance on some visualization best-practice questions we had.

### Matthew Goldsberry

During the project, I leveraged Claude Code as an assistant. It primarily helped me accelerate the generation of some of the visual styling by writing CSS based on provided descriptions of vision and troubleshooting bugs that stumped me. I also leveraged it to help with some of hte logic required for aggregating data into the visualizations from the read in CSV data, such as with top phrases for each character. This allowed me to maintain a rapid development pace overall by dealing with these items that would normally be speed bumps.

Additionally, Gemini was used to construct some assets in the project. Specifically the character icons, favicon, and image in the repo where generated using Gemini Nano Banana. Then the character descriptions and scene summaries where originally generated via Gemini before being annotated by myself. This also served the purpose of rapidly getting this information in with the given time constraints.

I did not receive any non-AI help from outside this team during the project.

### Isaac Dowdy

...

## Team Contributions

### Matthew Goldsberry

* ...

### Isaac Dowdy

* ...

### Joint Efforts

* ...
