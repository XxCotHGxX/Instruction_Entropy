(function () {
  let countries;
  let scores;
  let scoresMap;

  let geojsonLayer;
  let scrollContainer;
  let infoBox;
  const tooltip = d3.select("body").append("div").attr("id", "tooltip");

  const countryColWidth = 160;
  const scoreColWidth = 50;
  const tableRowHeight = 30;
  const factorBarHeight = 20;
  let factorColWidth;

  const maxScores = {};
  const scoreThresholds = [4, 5, 6, 7];
  const scoreColors = ["#d0587e", "#db8b95", "#e5b9ad", "#74ada2", "#009392"];
  const factorColors = [
    "#f3cbd3",
    "#eaa9bd",
    "#dd88ac",
    "#ca699d",
    "#b14d8e",
    "#91357d",
    "#6c2167",
  ].reverse();

  const tableColNames = ["Country", "Score", "All Factors"];
  const factors = [
    "Explained by: GDP per capita",
    "Explained by: Social support",
    "Explained by: Healthy life expectancy",
    "Explained by: Freedom to make life choices",
    "Explained by: Generosity",
    "Explained by: Perceptions of corruption",
    "Dystopia + residual",
  ];

  const formatScore = d3.format(".3f");

  const mapColorScale = d3
    .scaleThreshold()
    .domain(scoreThresholds)
    .range(scoreColors);

  const tableColorScale = d3.scaleOrdinal().domain(factors).range(factorColors);

  const tableColScale = d3.scaleOrdinal().domain(tableColNames);

  Promise.all([d3.json("countries.json"), d3.csv("whr-scores.csv")]).then(
    ([json, csv]) => {
      csv.forEach((d, i) => {
        csv.columns.forEach((column) => {
          if (column === "Country") return;
          d[column] = +d[column];
        });
        d.Rank = i + 1;
      });
      scores = csv;
      scoresMap = d3.map(csv, (d) => d.Country);

      countries = json;
      countries.features.forEach((el) => {
        const country = scoresMap.get(el.properties.name);
        if (country) {
          el.properties.score = country["Ladder score"];
        }
      });

      ["Ladder score"].concat(factors).forEach((factor) => {
        const maxScore = d3.max(scores, (d) => d[factor]);
        maxScores[factor] = maxScore;
      });

      renderMap();
      renderTable();
      updateInfo();
    }
  );

  window.addEventListener("resize", resizeTable);

  function renderMap() {
    const map = L.map("map", {
      scrollWheelZoom: false,
    }).setView([40, 0], 2);
    geojsonLayer = L.geoJson(countries, {
      style: style,
      onEachFeature: onEachFeature,
    }).addTo(map);

    const legend = L.control({ position: "bottomright" });
    legend.onAdd = drawLegend;
    legend.addTo(map);

    const info = L.control({ position: "bottomleft" });
    info.onAdd = () => L.DomUtil.create("div", "map-info");
    info.addTo(map);
    infoBox = document.getElementsByClassName("map-info")[0];

    function style(feature) {
      const score = feature.properties.score;
      return {
        fillColor: score ? mapColorScale(score) : "#ddd",
        weight: 1,
        color: "#fff",
        fillOpacity: 0.7,
      };
    }

    function drawLegend() {
      const div = d3.create("div").attr("class", "map-legend");

      div.append("div").attr("class", "legend-title").text("Ladder score");

      const legendLi = div
        .append("div")
        .attr("class", "legend-scale")
        .append("ul")
        .attr("class", "legend-labels")
        .selectAll("li")
        .data(scoreColors)
        .enter()
        .append("li");
      legendLi
        .append("span")
        .style("background-color", (d) => d)
        .style("opacity", 0.7);
      legendLi.append("span").text((d, i) => {
        if (i === 0) {
          return "<" + scoreThresholds[0];
        } else if (i === scoreColors.length - 1) {
          return ">" + scoreThresholds[scoreColors.length - 2];
        } else {
          return scoreThresholds[i - 1] + "-" + scoreThresholds[i];
        }
      });

      return div.node();
    }

    function onEachFeature(feature, layer) {
      if (!feature.properties.score) return;
      layer.on({
        mouseover: highlightFeature,
        mouseout: resetHighlight,
      });
    }
  }

  function renderTable() {
    const tableWidth = document.getElementById("table").clientWidth;
    factorColWidth = tableWidth - countryColWidth - scoreColWidth - 20;
    tableColScale.range([countryColWidth, scoreColWidth, factorColWidth]);

    const table = d3.select("#table");

    const th = table
      .append("thead")
      .append("tr")
      .selectAll("th")
      .data(tableColNames)
      .enter()
      .append("th")
      .style("width", (d) => tableColScale(d) + "px")
      .append("div")
      .text((d) => d);

    const tr = table
      .append("tbody")
      .selectAll("tr")
      .data(scores)
      .enter()
      .append("tr")
      .style("height", tableRowHeight + "px");

    tr.append("td")
      .attr("class", "country-label")
      .style("width", countryColWidth + "px")
      .text((d) => d.Country);

    tr.append("td")
      .attr("class", "country-score")
      .style("width", scoreColWidth + "px")
      .text((d) => formatScore(d["Ladder score"]));

    tr.append("td")
      .attr("class", "country-factor")
      .style("width", factorColWidth + "px")
      .append("div")
      .attr("class", "factor-wrapper")
      .style("width", factorColWidth + "px")
      .style("height", factorBarHeight + 2 + "px") // Account for top and bottom borders
      .selectAll(".factor-bar")
      .data((d) =>
        factors.map((factor) => ({
          factor: factor,
          value: d[factor],
        }))
      )
      .enter()
      .append("div")
      .attr("class", "factor-bar")
      .style("background-color", (d) => tableColorScale(d.factor))
      .style("width", (d) => (d.value / maxScores["Ladder score"]) * 100 + "%")
      .style("height", factorBarHeight + "px")
      .on("mouseover", showTooltip)
      .on("mouseout", hideTooltip);

    scrollContainer = table.select("tbody");

    const dropdown = th
      .filter((d, i) => i === tableColNames.length - 1)
      .attr("class", "ui dropdown");

    const dropdownFirstOption = dropdown.text();
    dropdown.text("");
    dropdown.append("div").attr("class", "text").text(dropdownFirstOption);
    dropdown.append("i").attr("class", "dropdown icon");

    dropdown
      .append("div")
      .attr("class", "menu")
      .selectAll("div")
      .data(["All Factors"].concat(factors))
      .enter()
      .append("div")
      .attr("class", (d, i) => {
        return i === 0 ? "active item" : "item";
      })
      .text((d) => d);

    $(".ui.dropdown").dropdown({
      onChange: factorChanged,
    });
  }

  function resizeTable() {
    const tableWidth = document.getElementById("table").clientWidth;
    factorColWidth = tableWidth - countryColWidth - scoreColWidth - 20;
    tableColScale.range([countryColWidth, scoreColWidth, factorColWidth]);

    const table = d3.select("#table");

    table.selectAll("th").style("width", (d) => tableColScale(d) + "px");
    table
      .selectAll(".country-factor")
      .style("width", factorColWidth + "px")
      .select(".factor-wrapper")
      .style("width", factorColWidth + "px");
  }

  function highlightFeature(e) {
    const layer = e.target;
    layer.setStyle({
      color: "#666",
    });

    if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
      layer.bringToFront();
    }

    const country = layer.feature.properties.name;
    scrollToRow(country);

    updateInfo(country);
  }

  function resetHighlight(e) {
    geojsonLayer.resetStyle(e.target);

    updateInfo();
  }

  function scrollToRow(country) {
    const countryIndex = scores.findIndex((d) => d.Country === country);
    if (countryIndex === -1) return;
    const scrollTop = countryIndex * tableRowHeight;
    scrollContainer
      .transition()
      .duration(1000)
      .tween("scrollTop", scrollTopTween(scrollTop));
  }

  function updateInfo(country) {
    if (!country) {
      infoBox.innerHTML = "<b>Hover over a country</b><br>Rank: <br>Score: ";
    } else {
      const d = scoresMap.get(country);
      infoBox.innerHTML = `<b>${d.Country}</b><br>Rank: <b>${
        d.Rank
      }</b><br>Score: <b>${formatScore(d["Ladder score"])}</b>`;
    }
  }

  function showTooltip(d) {
    d3.select(this).style("opacity", 1);

    tooltip.html(`${d.factor}<br>${formatScore(d.value)}`);
    const tooltipBCR = tooltip.node().getBoundingClientRect();
    const barBCR = this.getBoundingClientRect();

    tooltip
      .style("top", window.scrollY + barBCR.top - tooltipBCR.height - 6 + "px")
      .style(
        "left",
        window.scrollX +
          barBCR.left +
          barBCR.width / 2 -
          tooltipBCR.width / 2 +
          "px"
      )
      .style("opacity", 1);
  }

  function hideTooltip() {
    d3.select(this).style("opacity", 0.7);

    tooltip.style("opacity", 0);
  }

  function factorChanged(_, factor) {
    const table = d3.select("#table");

    table
      .selectAll(".country-score")
      .text((d) =>
        factor === "All Factors"
          ? formatScore(d["Ladder score"])
          : formatScore(d[factor])
      );

    table
      .selectAll(".factor-bar")
      .style("display", (d) => {
        if (factor === "All Factors") return "inline-block";
        return d.factor === factor ? "inline-block" : "none";
      })
      .style("width", (d) => {
        if (factor === "All Factors") {
          return (d.value / maxScores["Ladder score"]) * 100 + "%";
        } else {
          return (d.value / maxScores[factor]) * 100 + "%";
        }
      });
  }

  function scrollTopTween(scrollTop) {
    return function () {
      const i = d3.interpolateNumber(this.scrollTop, scrollTop);
      const node = this;
      return function (t) {
        node.scrollTop = i(t);
      };
    };
  }
})();
