import solarsystem
H = solarsystem.Heliocentric(year=2020, month=1, day=1, hour=12, minute=0 )
planets_dict=H.planets()
print('Planet','   \t','Longitude','   \t','Latitude','   \t','Distance in AU')
for planet in planets_dict:
    pos=planets_dict[planet]
    print(planet,'   \t',round(pos[0],2),'   \t',round(pos[1],2),'   \t',round(pos[2],2))