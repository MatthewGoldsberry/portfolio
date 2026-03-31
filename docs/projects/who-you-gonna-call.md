# Who You Gonna Call? 3-1-1!: Visual Disorder in Cincinnati

**Team Members:** [Matthew Goldsberry](https://github.com/MatthewGoldsberry) & [Isaac Dowdy](https://github.com/isaac-dowdy)

**Links:** [Live Application](https://visual-disorder-in-cincinnati.vercel.app/) | [Source Code (GitHub)](https://github.com/MatthewGoldsberry/Who-you-gonna-call) | [Demo Video](#video-demonstration)

---

## Project Overview & Motivations

TODO

## Video Demonstration

<figure markdown="span">
  <video controls loop muted playsinline width="700">
    <source src="" type="video/mp4"> 
    Your browser does not support the video tag.
  </video>
</figure>

## The Data

TODO

## Design Process & Early Sketches

At the start of the project, we established a requirement ourselves: the application must function as a single view with no scrolling. This constraint helps ensure that when a user interacts with a component, the filtering effects across all other visualizations are immediately visible to the user, without losing context.

### Initial Concept

The geographical interaction is the obvious driver of this data exploration, so the Leaflet map must be the centerpiece of the application. The secondary challenge to the layout was there were a lot of visualizations to add outside of just the Leaflet map, with 5 other bar charts and a timeline needing to be included. When designing the layout and estimating the sizing of the SVGs we had a strict mental requirement to ensure that all datapoints remained legible and easily intractable, making sure that no elements were to small to click or hover.

### Sketches

With these constraints in mind, we developed two early sketches to layout potential spatial arrangements. From this two primary layouts emerged:

#### Approach 1: Dual Chart Columns

![Approach 1 Sketch](../assets/media/who-you-gonna-call/sketch_1.png)

This approach surrounds the central Leaflet map with visualizations, dividing the bar charts across both the left and right margins. While this approach maximizes the total screen area dedicated to the charts, it crows the center and constrains the map's overall width.

#### Approach 2: Single Chart Column

![Approach 2 Sketch](../assets/media/who-you-gonna-call/sketch_2.png)

This approach consolidates all 5 bar charts into a single column on the right. This dedicates a much larger block of space for the Leaflet map and a little larger space for the timeline tool at the bottom. *(Note: This sketch also includes an early annotation exploring the possibility of utilizing the bottom-left quadrant for chart overflow).*

#### Decision and Validation

After evaluating both options, we went with **Approach 2**. The reasoning behind this was surrounding optimizing the spatial geometry:

* **More Ideal Aspect Ratios:** Combining the bar charts into a single column provided a more rectangular bounding box that better suited the horizontal nature of bar charts.
* **Map Size:** The Leaflet map required as much space as we could comfortably give it because of the known additions to come with specific controls that would take up some of its space. Approach 2 offered the largest amount of space to the map.

This was validated during implementation as once the map controls were added, they consumed significant screen real space, proving to some degree that the dual column support would have been too cramped.

We also noted during this sketching phase that the bar charts in the right column would be relatively small. To solve this in the final build, we introduced a chart-swapping feature that lets users move any bar chart into the large central map space for enhanced viewing.

## Visual Components & Interactions

TODO

## Key Discoveries & Findings

TODO

## Technical Implementation

TODO

## Challenges & Future Work

TODO

## Acknowledgements & AI Usage

TODO

## Team Contributions

TODO
