import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import warnings
import random

warnings.filterwarnings("ignore", message="PyplotGlobalUseWarning")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="EST PROJECT", page_icon=":rocket:", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply custom CSS styles
st.markdown(
    """
    <style>
    h1 {
        color: #2e7d32;
    }
    p {
        font-size: 16px;
    }
    .widget-label {
        color: #1976D2;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Create a navigation bar
pages = ["Introduction",
         "Worlds Annual CH₄ Emisssions (MCM)", 
         "Global Coal Mine Tracker", 
         "South African Registered coal mines", 
         "Locating Coal Mines", 
         "Top 15 coal producing Mines in South Africa",
         "Methane Emission from coal mines", 
         "Year wise Mines opened in South africa", 
         "Trend findings : Emission vs Year Count over years", 
         "Trend findings : Emission vs Coal Output", 
         "Thank you"]
selected_page = st.sidebar.radio("Select Page", pages)
df_main_operating = pd.read_excel('./Est/filtered-coal-South-Africa-operating.xlsx')
# Display content based on the selected page



if selected_page == "Introduction":
    institute_image = "./Est/iiit_logo.png"  # Replace with the actual image file path
    st.image(institute_image, use_column_width=True)
    st.title("Environmental Impact of Coal Mining in South Africa with a Focus on Methane Emissions")

    
     # Create columns for each person
    col1, col2 = st.columns(2)

    with col1:
        st.header("Advisor")
        st.subheader("Ramachandra Prasad P.")
        st.write("Associate Professor")
        st.write(" ***IIIT-Hyderabad***")
        advisor_papers = "[Advisor's Profile](https://www.iiit.ac.in/people/faculty/pramachandra/)"
        st.markdown(advisor_papers, unsafe_allow_html=True)

    with col2:
        team_members = ['LOKESH SHARMA', 'KRATI AGRAWAL ', 'HRISHIKESH DESHPANDE', 'ANUJA GULHANE', 'GAURAV KHAPEKAR']
        rolls = [2022201041, 2022201063, 2022201065, 2022201070, 2022201055]
        stream_team = "````PRAKRITI ````"
        university_name = "IIT University"
        professor_name = "Professor K"
        # Pair team members and rolls, shuffle the pairs
        team_info = list(zip(team_members, rolls))
        random.shuffle(team_info)

        # Unpack the shuffled pairs
        shuffled_team_members, shuffled_rolls = zip(*team_info)

        # Generate the stream text
        st.subheader(f"`Team-24` **{stream_team}**\n\n")
        stream_text = ""
        for member, roll in zip(shuffled_team_members, shuffled_rolls):
            stream_text += f"- {member.capitalize()} (Roll {roll})\n"

        stream_text += "\nWe have worked collaboratively to explore and analyze various aspects of our project, and we believe that our findings will contribute positively to the field.\n\n"
        # Print or use the 'stream_text' variable as needed
        st.write(stream_text)


elif selected_page == "Worlds Annual CH₄ Emisssions (MCM)":
    st.title("Worlds Annual CH₄ Emisssions (MCM)")
    def printEmisssionsData():
            # Streamlit app formatting
        st.title("Climate Change and Emissions Reduction: A Global Perspective")

        # Statement 1
        st.subheader("Concerns about Global Climate Discourse")
        st.write("Hypocrisy is evident when discussions about emissions reduction do not adequately address the contributions "
                "of major emitters like China and other developed countries.")

        # Statement 2
        st.subheader("Challenges Faced by Developing Nations")
        st.write("Developing nations often find it challenging to balance the need for emissions reduction with the imperative "
                "of economic growth. The struggle lies in addressing poverty and improving living standards while mitigating "
                "greenhouse gas emissions.")

        # Statement 3
        st.subheader("Call for Leadership by Developed Nations")
        st.write("Developed nations, with greater resources and technological advancements, have the opportunity to lead by "
                "example. They could play a crucial role in adopting and promoting sustainable practices globally, setting a "
                "positive precedent for others to follow.")
    def plot_top_emission_countries(data, column_name, top_n=10):
        """
        Plot a bar chart of the top countries with the highest values in the specified column.

        Args:
            data (DataFrame): The DataFrame containing the data.
            column_name (str): The column to use for sorting and plotting.
            top_n (int): The number of top countries to display (default is 10).
        """
        # Exclude 'TOTAL' row from the DataFrame
        data = data[data['Country'] != 'TOTAL']

        # Sort the DataFrame by the specified column in descending order
        data = data.sort_values(by=column_name, ascending=False)

        # Select the top N countries
        data_top = data.head(top_n)

        # Plotly bar chart
        fig = go.Figure()

        # Add traces for each country
        for index, row in data_top.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Country']],
                y=[row[column_name]],
                name=row['Country']
            ))

        # Update layout for better interactivity
        fig.update_layout(
            barmode='stack',
            xaxis=dict(title='Country'),
            yaxis=dict(title=column_name),
            title=f"Top {top_n} Countries by {column_name} Million cubic meters",
            showlegend=True,
            height=500
        )

        # Show the chart using Streamlit
        st.plotly_chart(fig)

    # Example usage:
    # Assuming you have a DataFrame named 'emission_data'
    file = './Est/AllCountries_Methane_Emission.xlsx'
    methane_emissoindf = pd.read_excel(file, sheet_name='Sheet1')
    # Customizable top_n using Streamlit slider
    top_n = st.slider("Select Top N Countries", min_value=1, max_value=15, value=10)
    plot_top_emission_countries(methane_emissoindf, "Annual CH₄ Emisssions (MCM)", top_n)
    printEmisssionsData()


elif selected_page == "Global Coal Mine Tracker":
    ## Global Coal Mine Tracker
    def printTrackerInfo():
        st.header("Global Coal Mine Tracker")
        st.markdown(
            """
            We are working on this data provided by *Global Coal Mine Tracker*, 
            sourced from *Global Energy Monitor*, October 2023 release.
            """
        )
        st.write(
            """
            The dataset contains a list of world coal mines, and our focus is on this site. 
            We are utilizing their data for our analysis.
            """
        )
        st.markdown(
            """
            [To download May 2023 data, click here](https://globalenergymonitor.org/wp-content/uploads/2023/06/Global-Coal-Project-Finance-Tracker_May-2023_Final.xlsx)
            """
        )
        st.markdown(
            """
            [To download October 2023 data, click here](https://globalenergymonitor.org/wp-content/uploads/2023/10/Global-Coal-Mine-Tracker-October-2023.xlsx)
            """
        )
    
    def printMinesDetails(df):
            # Read the Excel file
        # Display total coal mines and columns
        st.write(f'Total Registered coal mines in ```AFRICAN Region```: {df.shape[0]}')
        st.write(f'Total columns: {df.shape[1]}')
        # Display the first 2 rows of the DataFrame
        st.dataframe(df.sample(5))


        st.title(" Separating Africa from all list")
        st.write("MAJOR FOCUS - AFRICA CONTINENT")
        st.header("Why South Africa?")
    def showMinesAfrica(counts):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color='blue',  # You can customize the color
        ))

        # Update layout for better visualization
        fig.update_layout(
            title='Number of Registered Coal Mines in African Countries',
            xaxis=dict(title='Country'),
            yaxis=dict(title='Number of Mines'),
        )

        # Show the chart using Streamlit
        st.plotly_chart(fig)

    def MineChart(df):
        df_region = df[df['Region'].str.contains('AFRICA', case=False)]
        st.write(f' Total ```African``` Registered coal mines = {df_region.shape[0]}\n')
        df_africa = df[df['Country'].str.contains('AFRICA', case=False)]
        st.write(f' Total ```South African``` Registered coal mines = {df_africa.shape[0]}')

        counts = df['Country'].value_counts()
        st.header('List of African countries with number of  registered coal mines')
        st.dataframe(counts.to_frame().T)
        showMinesAfrica(counts.T)
    printTrackerInfo()
    df = pd.read_excel('./Est/world-coal-mines-october23.xlsx')
    st.write(f'Total Registered coal mines in the ```WORLD```: {df.shape[0]}')
    st.write(f'Total columns: {df.shape[1]}')
    st.dataframe(df.sample(5))
    st.subheader("Separating African Region")
    df = pd.read_excel('./Est/coal-Africa.xlsx')
    printMinesDetails(df)
    MineChart(df)
    # Assuming 'df' is your DataFrame and 'Subregion' is the column you want to filter
    

elif selected_page == "Methane Emission from coal mines":
    
    st.title("Methane Emission from coal mines")
    st.write("- Methane emissions from coal mines primarily result from the release of methane gas trapped within the coal seams. Coal is composed of organic matter, and as the plant material that forms coal undergoes geological processes over millions of years, methane (CH₄) is generated as a byproduct. This process is known as coal bed methane (CBM) generation.")
    st.write("- The methane produced during coal formation becomes trapped within the coal seams, either adsorbed onto the coal particles or within fractures and pores in the coal.")
    st.write("- When coal is mined, the pressure that keeps the methane trapped is reduced, allowing the gas to be released. This release of methane during mining is commonly referred to as 'coal mine methane' (CMM) emissions.")
    st.write("- At a typical gassy coal mine, ventilation air may contain 0.1 to 1% methane, whereas gas drained from the seam before mining can contain 60% to more than 95% methane.")
    st.write("- On average, it is estimated that methane is responsible for 9.6% to 23% of a coal mine’s greenhouse gas (GHG) emissions.")
    st.write("- Underground mines typically emit more methane than surface mines since methane content increases with pressure and depth.")
    st.write("- Underground mines contributed 85% (44 Mt) of total coal mine methane emissions, while surface and mixed mines accounted for 15% (8 Mt).")
    st.write("- South Africa (ranked 5th), a country that relies primarily on underground mining, emits more methane than the world’s second largest coal producer India (ranked 6th), which relies primarily on surface mining.")
    st.header("Factors contributing to Methane emission:")
    st.markdown("1. Coal Composition:")
    st.write("- The methane content in coal varies depending on the type of coal and its maturity. Bituminous and sub-bituminous coals generally contain higher methane levels compared to anthracite coal.")
    df = pd.read_excel('./Est/coal_type.xlsx')
    st.dataframe(df)
    st.markdown("2. Methane Content:")
    st.write("- The amount of methane released when one metric tonne of coal is used depends on the type of coal. Here is a table of the approximate methane content and methane emission percentage for different types of coal:")
    df = pd.read_excel('./Est/coal_CH4.xlsx')
    st.dataframe(df)
    st.write("- Overall, the ``` methane``` emission percentage is a relatively small percentage of the total ```carbon dioxide emission``` percentage when coal is burned. ``However, because methane is such a potent greenhouse gas, even small emissions can have a significant impact on climate change.``")

    st.header("Things to do in  order to reduce methane emission:")
    st.write("- Use cleaner types of coal, such as anthracite.")
    st.write("- Capture methane emissions from coal mines and use them as a fuel source or for other purposes.")
    st.write("- Transition to cleaner energy sources, such as renewable energy like wind energy, solar energy")

elif selected_page == "South African Registered coal mines":
    def SAmines(df_africa):
        st.title("South African Registered coal mines")
        st.write(f' Total **South African**  Registered coal mines = {df_africa.shape[0]}')
        st.dataframe(df_africa.sample(5))
        st.write("See specific information is only available for currently ```operating mines```.")
        st.write("Step by step process to get the data")
        data_extraction_process = "[Data_extraction_process](https://github.com/Lokeshiiith/EstProject/blob/main/est.ipynb)"
        st.markdown(data_extraction_process, unsafe_allow_html=True)
    def SA_OperatingMines(df):
        st.header("South African Coal Mines Which are currently at ```operating``` status")
        df = df[df['Status'].str.contains('Operating', case=False)]
        df = df.dropna(subset=['Opening Year'])
        #REMOVING THOSE ROWS WHICH HAVE 'TBD' IN THE OPENING YEAR COLUMN
        st.write(f'Total South African Registered coal mines with Status is operating and a known Opening Year = {df.shape[0]}')
    df = pd.read_excel('./Est/coal-south-africa.xlsx')
    SAmines(df)
    
    SA_OperatingMines(df_main_operating)
    st.dataframe(df_main_operating.sample(10))

elif selected_page == "Locating Coal Mines":
    def show_map(df_operating):
        # Create the map as before
        m = folium.Map(location=[10, 15], zoom_start=3)

        south_africa_geo = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [21.002324, -34.821182],
                                [32.882409, -34.821182],
                                [32.882409, -22.137945],
                                [21.002324, -22.137945],
                                [21.002324, -34.821182],
                            ]
                        ]
                    }
                }
            ]
        }

        def style_function(feature):
            return {
                'fillColor': 'yellow',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.3
            }

        folium.GeoJson(
            south_africa_geo,
            name='South Africa',
            style_function=style_function
        ).add_to(m)

        # Add markers for each mine
        for index, row in df_operating.iterrows():
            try:
                emissions_estimate = int(row['Coal Mine Methane Emissions Estimate (MCM/yr)'])
            except (ValueError, TypeError):
                emissions_estimate = 10
            
            if emissions_estimate >= 80:
                marker_color = 'red'
            elif emissions_estimate >= 50:
                marker_color = 'orange'
            else:
                marker_color = 'blue'
            
            tooltip_text = f"{row['Mine Name']} - Emissions: {emissions_estimate} MCM/yr"
            
            marker = folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                tooltip=tooltip_text,
                icon=folium.Icon(color=marker_color)
            )
            
            marker.add_to(m)

        # Display the map using folium_static
        folium_static(m)


    st.title("Locating Coal Mines")
    df = pd.read_excel('./Est/filtered-coal-Africa.xlsx')
    df = df[df['Status'].str.contains('Operating', case=False)]
    df = df.dropna(subset=['Opening Year'])
    df_operating = df[df['Opening Year'] != 'TBD']
    #REMOVING THOSE ROWS WHICH HAVE 'TBD' IN THE OPENING YEAR COLUMN
    st.write(f'Total South African Registered coal mines with Status is operating and a known Opening Year = {df.shape[0]}')
    st.write("- Red ones are the mines with high Methane emissions")
    st.write("- Orange ones are the mines with medium Methane emissions")
    st.write("- Blue ones are the mines with low Methane emissions")
    show_map(df_operating)
    # Drop specific columns
    df_operating = df_main_operating
    columns_to_drop = ['Status', 'Coal Output (Annual, Mt)', 'Country']
    df_operating = df_operating.drop(columns=columns_to_drop)
    # Display the DataFrame in Streamlit
    df_operating['Coal Mine Methane Emissions Estimate (MCM/yr)'] = pd.to_numeric(df_operating['Coal Mine Methane Emissions Estimate (MCM/yr)'], errors='coerce').fillna(0)
    st.dataframe(df_operating)
    # Convert "Coal Mine Methane Emissions Estimate" to float and fill NaN with 0




elif selected_page == "Top 15 coal producing Mines in South Africa":
    st.title("Top 15 coal producing Mines in South Africa")
    def ShowTop10(df):
    # Filter out rows with missing coal output values
            # Filter out rows with missing coal output values
        df['Coal Output (Annual, Mt)'] = pd.to_numeric(df['Coal Output (Annual, Mt)'], errors='coerce')

        data = df.dropna(subset=['Coal Output (Annual, Mt)'])
        
        # Sort the DataFrame by coal output in descending order
        data = data.sort_values(by='Coal Output (Annual, Mt)', ascending=False)
        
        # Select the top 10 coal mines
        top_10_mines = data.head(15)
        
        # Create a Folium map
        m = folium.Map(location=[0, 0], zoom_start=3)

        # Add markers for the top 10 coal mines
        for index, row in top_10_mines.iterrows():
                try:
                    emissions_estimate = int(row['Coal Mine Methane Emissions Estimate (MCM/yr)'])
                except (ValueError, TypeError):
                    emissions_estimate = 0  # Set to 0 if conversion fails
                if emissions_estimate >= 100:
                    marker_color = 'red'
                else:
                    marker_color = 'blue'
                tooltip_text = f"Name: {row['Mine Name']}<br>Coal Output(Annual, Mt): {row['Coal Output (Annual, Mt)']} <br>Year: {row['Opening Year']}<br>Methane Emissions (MCM/yr): {emissions_estimate}"
                tooltip_text += f'<br>Coal Type: {row["Coal Type"]}'
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    tooltip=tooltip_text,
                    icon=folium.Icon(color=marker_color)
                ).add_to(m)

        # Display the Folium map in Streamlit using st.pydeck_chart
        folium_static(m)
    df_operating = df_main_operating
    ShowTop10(df_operating)
   # Sample observations
    observations = [
        "The Grootegeluk coal mine is an open-pit coal mine located in Limpopo Province, South Africa.",
        "It is owned and operated by Exxaro Resources, one of the largest coal producers in South Africa.",
        "The mine has been in operation since 1980 and produces about 26 million tonnes of coal per year.",
        "Environmental impact: Major source of methane emissions.",
        "Future: Could close by 2050."
    ]
    # Display observations as bullet points
    st.title("Observations")
    st.write("- " + "\n- ".join(observations))


elif selected_page == "Year wise Mines opened in South africa":
    st.title("Year wise Mines opened in South africa")
    def showPlot(df_operating):
        df_operating['Opening Year'] = pd.to_numeric(df_operating['Opening Year'], errors='coerce')

        mine_counts = df_operating['Opening Year'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        ax = mine_counts.plot(kind='bar', xlabel='Opening Year', ylabel='Number of Mines opened', title='Mines Opened by Year')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        # Add values on top of the bars
        for i, v in enumerate(mine_counts):
            ax.text(i, v, str(v), ha='center', va='bottom')

        plt.tight_layout()

        # Display the bar chart using st.pyplot
        st.pyplot(plt)
        st.dataframe(df_operating.sample(5))
    def showMethaneEmissionYearWisePlot(df):
        df = df_main_operating
        df['Coal Mine Methane Emissions Estimate (MCM/yr)'] = pd.to_numeric(df['Coal Mine Methane Emissions Estimate (MCM/yr)'], errors='coerce')
        # Group and accumulate emissions by year
        # Group and accumulate emissions by year
        emissions_by_years = df.groupby('Opening Year')['Coal Mine Methane Emissions Estimate (MCM/yr)'].sum().reset_index()

        # Create a line graph using Plotly
        fig = px.line(emissions_by_years, x='Opening Year', y='Coal Mine Methane Emissions Estimate (MCM/yr)',
                    labels={'Coal Mine Methane Emissions Estimate (MCM/yr)': 'Total Emissions (MCM/yr)'},
                    title='Coal Mine Methane Emissions by Year')

        # Add hover data (details)
        fig.update_traces(hovertemplate='Year: %{x}<br>Total Emissions: %{y} MCM/yr')

        # Show the chart using Streamlit
        st.plotly_chart(fig)
    df = pd.read_excel('./Est/coal-Africa.xlsx')
    showPlot(df)
    showMethaneEmissionYearWisePlot(df)


elif selected_page == "Trend findings : Emission vs Year Count over years":
    def showEmissionVSNewMinesPlot1(emissions_by_years, new_mines_opened_byYear):
        df_emissions = pd.DataFrame(emissions_by_years)
        df_mine_count = pd.DataFrame(new_mines_opened_byYear)

        # Combine both DataFrames based on the 'Year' column
        combined_df = df_emissions.merge(df_mine_count, on='Year')

        # Create a Plotly figure using px.scatter
        fig = px.scatter(combined_df, x='Year', y='Methane Emission', color_discrete_sequence=['blue'], labels={'Methane Emission': 'Methane Emission'})
        
        # Add a vertical line at year 2004
        fig.add_shape(type='line', x0=2004, x1=2004, y0=0, y1=1, line=dict(color='gray', dash='dash'))

        # Create a second y-axis for Mine Count
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'), yaxis2_title_text='Mine Count')

        # Add Mine Count data to the figure
        fig.add_trace(px.scatter(combined_df, x='Year', y='Num of New Mines Opened', color_discrete_sequence=['red']).data[0])

        # Update layout for better readability
        fig.update_layout(xaxis_title='Year', yaxis_title='Methane Emission')

        # Show the chart using Streamlit
        st.plotly_chart(fig)
    def showEmissionVSNewMinesPlot2(emissions_by_years, new_mines_opened_byYear):
        # Create DataFrames from the sample data
        df_emissions = pd.DataFrame(emissions_by_years)
        df_mine_count = pd.DataFrame(new_mines_opened_byYear)

        # Combine both DataFrames based on the 'Year' column
        combined_df = df_emissions.merge(df_mine_count, on='Year')

        # Create a figure and axis
        fig, ax1 = plt.subplots(figsize=(8, 4))

        # Plot Methane Emission on the primary y-axis
        ax1.plot(combined_df['Year'], combined_df['Methane Emission'], marker='o', label='Methane Emission', color='tab:blue')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Methane Emission', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a secondary y-axis
        ax2 = ax1.twinx()

        # Plot Mine Count on the secondary y-axis
        ax2.plot(combined_df['Year'], combined_df['Num of New Mines Opened'], marker='o', label='Mine Count', color='tab:red')
        ax2.set_ylabel('Mine Count', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        # Add a vertical line at year 2004
        ax1.axvline(x=2004, color='gray', linestyle='--', label='MPDRA Introduced')
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('Emissions vs. Mine Count Over the Years')
        plt.grid(True)
        st.pyplot(plt)

    st.title("Trend findings : Emission vs Year Count over years")
    df_operating = df_main_operating[df_main_operating['Opening Year'] != 'TBD']
    df_operating['Coal Mine Methane Emissions Estimate (MCM/yr)'] = pd.to_numeric(df_operating['Coal Mine Methane Emissions Estimate (MCM/yr)'], errors='coerce')
    # Group and accumulate emissions by year
    emissions_by_years = df_operating.groupby('Opening Year')['Coal Mine Methane Emissions Estimate (MCM/yr)'].sum()
    new_mines_opened_byYear = df_operating.groupby('Opening Year')['Coal Mine Methane Emissions Estimate (MCM/yr)'].count()
    emissions_by_years = pd.DataFrame({'Year': emissions_by_years.index, 'Methane Emission': emissions_by_years.values})
    new_mines_opened_byYear = pd.DataFrame({'Year': new_mines_opened_byYear.index, 'Num of New Mines Opened': new_mines_opened_byYear.values})
    # showEmissionVSNewMinesPlot1(emissions_by_years, new_mines_opened_byYear)
    showEmissionVSNewMinesPlot2(emissions_by_years, new_mines_opened_byYear)
    st.write("- It seems to suggest that after the introduction of the MPDRA act in 2004, the number of new coal mines opened in South Africa increased. This is inferred from the trend in the ''' Num of New Mines Opened ''' data, especially after the vertical line at the year 2004.")



elif selected_page == "Trend findings : Emission vs Coal Output":
    st.write("Trend findings : Emission vs Coal Output")
    # Convert the 'Opening Year' column to integers
    def ShowEmissionvsCoalPlot(emissions_by_years, coal_output_by_year):
        # Create DataFrames from the sample data
        df_emissions = pd.DataFrame(emissions_by_years)
        df_coal_output_by_year = pd.DataFrame(coal_output_by_year)

        # Combine both DataFrames based on the 'Year' column
        combined_df = df_emissions.merge(df_coal_output_by_year, on='Year')

        # Create a figure and axis
        fig, ax1 = plt.subplots(figsize=(8, 4))

        # Plot Methane Emission on the primary y-axis
        ax1.plot(combined_df['Year'], combined_df['Coal Mine Methane Emissions Estimate (MCM/yr)'], marker='o', label='Coal Mine Methane Emissions Estimate (MCM/yr)', color='tab:blue')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Coal Mine Methane Emissions Estimate (MCM/yr)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a secondary y-axis
        ax2 = ax1.twinx()

        # Plot Coal Output on the secondary y-axis
        ax2.plot(combined_df['Year'], combined_df['Coal Output (Annual, Mt)'], marker='o', label='Coal Output', color='tab:red')
        ax2.set_ylabel('Coal Output (Annual, Mt)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        ax1.axvline(x=2004, color='gray', linestyle='--', label='MPDRA Introduced')
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('Methane Emissions vs. Coal Output (metric Ton)')
        plt.grid(True)
        st.pyplot(plt)
    
    df = df_main_operating  
    df_coalOutput = df
    df_coalOutput['Opening Year'] = pd.to_numeric(df_coalOutput['Opening Year'], errors='coerce')
    df_coalOutput['Coal Output (Annual, Mt)'] = pd.to_numeric(df_coalOutput['Coal Output (Annual, Mt)'], errors='coerce')
    coal_output_by_year = df_coalOutput.groupby('Opening Year')['Coal Output (Annual, Mt)'].sum()
    coal_output_by_year = pd.DataFrame({'Year': coal_output_by_year.index, 'Coal Output (Annual, Mt)': coal_output_by_year.values})
    df_operating = df 
    df_operating['Coal Mine Methane Emissions Estimate (MCM/yr)'] = pd.to_numeric(df_operating['Coal Mine Methane Emissions Estimate (MCM/yr)'], errors='coerce')
    emissions_by_years = df_operating.groupby('Opening Year')['Coal Mine Methane Emissions Estimate (MCM/yr)'].sum()
    emissions_by_years = pd.DataFrame({'Year': emissions_by_years.index, 'Coal Mine Methane Emissions Estimate (MCM/yr)': emissions_by_years.values})
    ShowEmissionvsCoalPlot(emissions_by_years, coal_output_by_year)


elif selected_page == "CO2 emission overs the years":
    st.title("Mean CO2 emission over the years")
    st.write("will update later .. .... ...")


