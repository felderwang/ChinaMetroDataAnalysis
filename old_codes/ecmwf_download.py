import cdsapi


def d_year(c, year):
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'fraction_of_cloud_cover', 'geopotential', 'ozone_mass_mixing_ratio',
                'relative_humidity', 'specific_cloud_liquid_water_content', 'specific_humidity',
                'specific_rain_water_content', 'temperature', 'u_component_of_wind',
                'v_component_of_wind', 'vertical_velocity',
            ],
            'pressure_level': [
                '875', '900', '925',
                '950', '975', '1000',
            ],
            'year': year,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '06:00', '12:00',
                '18:00',
            ],
            'area': [
                45, -90, 30,
                -70,
            ],
            'format': 'netcdf',
        },
        f'./datas/netcdf_nahg/nahg_year{year}.nc')
    
def main():
    c = cdsapi.Client()
    year_list = range(2010, 2021)
    print(f'year_list:{year_list}')
    for year in year_list:
        print(f'download {year}')
        d_year(c, str(year))
'''
guokong 
'area': [
                34, 115, 28,
                122,
            ],
nahg:
'area': [
                45, -90, 30,
                -70,
            ],
'''
if __name__ == '__main__':
    main()