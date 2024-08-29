import datetime
import pytz
import numpy as np
from astropy.time import Time
from astropy import coordinates
from astropy import units as u
from observing_suite import Target, ObservingPlan

dim_d = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
fmt = '%Y/%m/%d %I:%M %p'

class ObservingSite ( object ):
    '''
    [Determine airmass as a function of time for a given observing catalog
    and date.]
    Defines an observing site from which we may calculate observing conditions,
    given an target list and date.
    '''
    def __init__ ( self, site='CTIO', timezone=None):
        if site=='CTIO':
            self.site = coordinates.EarthLocation ( lat='-30d10m10.78s',
                                                    lon='-70d48m23.49s',
                                                    height=2200.*u.m )
            self.timezone= pytz.timezone ( 'America/Santiago' )
        elif site=='palomar':
            self.site = coordinates.EarthLocation ( lat='+33d21m22.7s',
                                                    lon='-116d51m53.6s',
                                                    height=1712.*u.m )
            self.timezone= pytz.timezone ( 'America/Los_Angeles' )            
        else:
            # // if not CTIO, trust the user to put in an EarthLocation
            # // and pytz.timezone ()
            self.site = site
            if type(timezone) == str:
                self.timezone = pytz.timezone(timezone)
            else:
                self.timezone = timezone

    def get_sunriseset ( self, year, month, day, alt=-14., cut_at_contract=False, contract_time=(6,5), return_track=False ):
        '''
        DECam observing begins and ends at Sun altitude=-14 deg.
        
        cut_at_contract: bool, default=False
            if True, cut at <contract_time> in accordance to Chilean labor laws
        contract_time: tuple, default=(6,5)
            hour,min in Chilean local time at which observations must cease
        '''
        print (f'[observe] Computing sunrise and sunset on {year}/{month}/{day} at altitude = {alt}' )
        utc_midnight = pytz.utc.localize ( datetime.datetime ( year, month, day, 0, 0 ) )
        utc_offset = int(self.get_utcoffset (utc_midnight))

        utc_start = pytz.utc.localize ( datetime.datetime ( year, month, day, 12-utc_offset, 0))
        
        if (day == dim_d[month]) & (month==12):
            print('happy new years')
            utc_end = pytz.utc.localize ( datetime.datetime ( year+1, 1, 1, 12-utc_offset,0) )
        elif day == dim_d[month]:
            utc_end = pytz.utc.localize ( datetime.datetime ( year, month+1, 1, 12-utc_offset,0) )
        else:
            utc_end = pytz.utc.localize ( datetime.datetime ( year, month, day+1, 12-utc_offset,0) )
        


        grid = np.arange(Time(utc_start), Time(utc_end),10.*u.min)
        fgrid = np.arange(Time(utc_start), Time(utc_end), 1.*u.min)
        sun_alt = []
        for ts in grid:
            sun_coord = coordinates.get_sun ( ts )
            obsframe = coordinates.AltAz ( obstime=ts, location=self.site )
            sun_alt.append( sun_coord.transform_to(obsframe).alt )

        sun_alt = np.asarray( [ sa.value for sa in sun_alt ] )
        if return_track:
            return grid, sun_alt
        
        fgrid_unix = np.asarray([ gg.unix for gg in fgrid ])
        grid_unix = np.asarray([ gg.unix for gg in grid ])
        
        sun_falt = np.interp(fgrid_unix, grid_unix, sun_alt)
        
        observable = sun_falt <= alt
        obs_can_start, obs_must_end = fgrid[observable][[0,-1]]
        
        lcl = lambda x: pytz.utc.localize(x.to_datetime())
        night_start = lcl(obs_can_start)
        night_end = lcl(obs_must_end)
        
        
        if cut_at_contract:            
            utcoff = self.get_utcoffset ( night_end )
            contract_end = pytz.utc.localize(datetime.datetime ( night_end.year, night_end.month, night_end.day, contract_time[0]-int(utcoff), contract_time[1],))
            mat_end = min ( night_end, contract_end )
            
            if contract_end < night_end:
                print ( 'True night end is:')                
                print(f"obsEnd:   {night_end.astimezone(self.timezone).strftime(fmt)} Santiago")        
                print(f"          {night_end.strftime(fmt)} UTC")
                                
                print ( 'Updated night end is:')
                print(f"obsEnd:   {contract_end.astimezone(self.timezone).strftime(fmt)} Santiago")        
                print(f"          {contract_end.strftime(fmt)} UTC")    
            return night_start, mat_end
        else:
            return night_start, night_end
    def track_moon ( self, start_time, end_time, alt_returntype='max' ):
        '''
        Over a starting and ending time, return the illumination (fractional) and 
        maximum altitude of the moon.
        '''
        from .astroplan_moon import moon_illumination
        
        time_start_time = Time(start_time)
        time_end_time = Time(end_time)
        
        grid = np.arange(time_start_time, time_end_time,10.*u.min)
        fgrid = np.arange(time_start_time, time_end_time, 1.*u.min)
        
        moon_alt = []
        for ts in grid:
            moon_coord = coordinates.get_moon ( ts )
            obsframe = coordinates.AltAz ( obstime=ts, location=self.site )
            moon_alt.append( moon_coord.transform_to(obsframe).alt )
        moon_alt = np.asarray ( [ ma.value for ma in moon_alt ])
        
        fgrid_unix = np.asarray([ gg.unix for gg in fgrid ])
        grid_unix = np.asarray([ gg.unix for gg in grid ])
        moon_falt = np.interp(fgrid_unix, grid_unix, moon_alt)        
        
        moon_altreport = getattr(np, alt_returntype)(moon_falt)
        moon_altreport = max(0., moon_altreport)
        
        moon_cillum = np.mean([moon_illumination (time_start_time), 
                               moon_illumination(time_end_time)])
        
        return moon_cillum, moon_altreport
        

    def get_utcoffset ( self, date ):
        '''
        From a datetime object, return the UTC offset at the observing site
        at that date/time.
        '''
        # // because of daylight savings time,
        # // we need to calculate the UTC offset *after* we know the
        # // observation date.
        # // WARNING ) pytz-2018 has an outdated handling of Chilean
        # // daylight savings time. Make sure your pytz version is up-to-date (2020).
        return date.astimezone ( self.timezone ).utcoffset().total_seconds()/3600.

    def define_obsframe ( self, obs_start=None, nstep=1., lim=6.,
                          obs_end=None ):
        '''
        For an individual night of an observing run, generate the
        observing frame. The observing frame specifies observatory location
        + time for +/- lim hours about the fiducial obs_datetime in steps of nstep
        hours.
        '''
        #utcoffset = self.get_utcoffset ( obs_datetime )

        utc_start = Time(obs_start) - obs_start.astimezone(pytz.utc).minute*u.minute - obs_start.astimezone(pytz.utc).second*u.second
        if obs_end is not None:
            utc_end = Time(obs_end) - obs_end.astimezone(pytz.utc).minute*u.minute -obs_end.astimezone(pytz.utc).second*u.second
        else:
            utc_end = utc_start + lim*u.hour

        
        #frame = np.arange ( -lim, lim+nstep/2., nstep) * u.hour
        #frame = np.arange(0, (utc_end-utc_start).to_value(u.hour)+1,1)*u.hour
        timeframe = np.arange(utc_start+0.5*u.hour, utc_end+1*u.hour, 1.*u.hour) #utc_start + frame
        #timeframe += 0.5*u.hour # calculate airmass in middle of hour
        obsframe = coordinates.AltAz ( obstime = timeframe, location=self.site)
        return obsframe

    def get_altitude ( self, target_coord, obsframe ):
        '''
        For an observing frame and target coordinates, get altitude as a
        function of time
        '''
        alt = target_coord.transform_to ( obsframe )
        return alt
    
    def utcoffset ( self, obsdate ):
        offset = self.timezone.utcoffset ( obsdate )
        offset_hrs = offset.seconds/3600. +  offset.days*24
        return offset_hrs

class TargetList (object):
    def __init__(self, coordlist, target_names ):
        if not isinstance(coordlist, coordinates.SkyCoord ):
            coordlist = coordinates.SkyCoord(coordlist, unit='deg')
        
        self.coordinates = coordlist
        self.target_names = target_names
        #self.target_types = target_types

class PalomarTargetList ( TargetList ):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.obssite = ObservingSite('palomar')
    
    def make_target_list ( self, obsdates, slit_width, offset_stars=None, savename=None ):
        target_list_obs_suite = []
        
        for idx in range(len(self.coordinates)):        
            custom_target = Target(
                name=self.target_names[idx],
                coordinates=self.coordinates[idx],
                parse_name=False
            )
            custom_target.add_configuration(
                config_name='primary',
                obstype='spectroscopy',
                slit_width=slit_width
            )
            
            if offset_stars is not None:
                if offset_stars[idx] is not None:
                    custom_target.add_offset_star(
                        coordinate=offset_stars[idx],
                        configurations='all'
                    )
            
            target_list_obs_suite.append(custom_target)
            
        if isinstance(obsdates, list):
            obsdate = obsdates[0]
        else:
            obsdate = obsdates 
            
        plan = ObservingPlan(
            target_list_obs_suite,
            observatory='Palomar',
            obsdates = obsdates,
            utcoffset=self.obssite.utcoffset ( datetime.datetime.strptime('2024-09-05', '%Y-%m-%d') )
        )
        
        if savename is None:
            savename = f'palomar_targets_%s' % obsdate
        plan.export_targetlist(name=savename, include_extras=['offsets'])
        return plan