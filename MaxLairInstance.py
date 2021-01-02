#   MaxLairInstance
#       Eric Donders
#       2020-11-20

import cv2, time, pytesseract, enchant, pickle, sys
from datetime import datetime
from typing import TypeVar, Dict, List, Tuple
Pokemon = TypeVar('Pokemon')
Move = TypeVar('Move')
Serial = TypeVar('serial.Serial')
VideoCapture = TypeVar('cv2.VideoCapture')
DateTime = TypeVar('datetime.datetime')
Image = TypeVar('cv2 image')

class MaxLairInstance():
    def __init__(self,
                 boss: str,
                 balls: int,
                 com: Serial,
                 cap: Serial,
                 lock,
                 exit_flag,
                 datetime: DateTime,
                 pokemon_data_paths: Tuple[str, str, str, str],
                 mode: str,
                 dynite_ore: int,
                 stage: str='join') -> None:
        self.boss_pokemon_path, self.rental_pokemon_path, self.boss_matchups_path, self.rental_matchups_path, self.rental_scores_path, self.items_path = pokemon_data_paths
        self.reset_run()
        
        self.start_date = datetime
        self.filename = ''.join(('Logs//',boss,'_',datetime.strftime('%Y-%m-%d %H-%M-%S'),'_log.txt'))
        self.boss = boss
        self.base_ball, self.base_balls, self.legendary_ball, self.legendary_balls = balls
        self.mode = mode
        self.dynite_ore = dynite_ore
        self.stage = stage

        self.runs = 0
        self.wins = 0
        self.shinies_found = 0
        self.consecutive_resets = 0

        #Additional variables
        self.shot = 0
        self.lives = 4
        self.caught_total = 0

        # Video capture and serial communication objects
        self.cap = cap
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.com = com
        self.lock = lock
        self.exit_flag = exit_flag

        # Rectangles for checking shininess and reading specific text
        # Shiny star rectangle
        self.shiny_rect = ((0.09,0.533), (0.12,0.573))
        # Selectable Pokemon names rectangles
        self.sel_rect_1 = ((0.485,0.295), (0.61,0.335))
        self.sel_rect_2 = ((0.485,0.545), (0.61,0.585))
        self.sel_rect_3 = ((0.485,0.790), (0.61,0.830))
        self.sel_rect_4 = ((0.485,0.590), (0.61,0.630))
        # In-battle Pokemon name & type rectangles
        self.sel_rect_5 = ((0.205,0.125), (0.400,0.185))
        self.type_rect_1 = ((0.255,0.189), (0.320,0.230))
        self.type_rect_2 = ((0.360,0.189), (0.430,0.230))
        # Selectable Pokemon abilities rectangles
        self.abil_rect_1 = ((0.485,0.340), (0.61,0.385))
        self.abil_rect_2 = ((0.485,0.590), (0.61,0.635))
        self.abil_rect_3 = ((0.485,0.835), (0.61,0.880))
        self.abil_rect_4 = ((0.485,0.635), (0.61,0.675))
        # Poke ball rectangle
        self.ball_rect = ((0.69,0.620), (0.88,0.665))
        self.ball_n_rect = ((0.890,0.620), (0.930,0.665))
        # Backpacker
        self.item_rect_1 = ((0.549,0.1270), (0.745,0.1770)) 
        self.item_rect_2 = ((0.549,0.2035), (0.745,0.2535))
        self.item_rect_3 = ((0.549,0.2800), (0.745,0.3300))
        self.item_rect_4 = ((0.549,0.3565), (0.745,0.4065))
        self.item_rect_5 = ((0.549,0.4330), (0.745,0.4830))

        #Items table
        self.tier_1_items = ['Restos',
                        'Cascabel Concha']
        self.tier_2_items = ['Vidasfera',
                        'Cinta Eleccion',
                        'Panuelo Eleccion',
                        'Gafas Eleccion',
                        'Periscopio']
        self.tier_3_items = ['Banda Focus',
                        'Cinta Experto',
                        'Gafas Especiales',
                        'Cinta Fuerte',
                        'Lupa']
        self.tier_4_items = ['Seguro Debilidad',
                        'Seguro Fallo',
                        'Chaleco Asalto',
                        'Gafa Protectora',
                        'Parasol Multiuso',
                        'Casco Dentado']
        self.tier_5_items = ['Baya Zanama',
                        'Baya Ziuela',
                        'Baya Zidra']

        #Base Ball selection table
        self.base_dive = ['Sandslash (Alolan Form)']
        self.base_lux = ['Dugtrio (Alolan Form)',
                         'Persian (Alolan Form)',
                         'Marowak (Alolan Form)',
                         'Weezing',
                         'Stunfisk']                            
        self.base_premier = ['Persian',
                           'Mr. Mime',
                           'Linoone']
        self.base_dream = ['Porygon']

    def reset_run(self) -> None:
        """Reset in preparation for a new Dynamax Adventure."""
        self.pokemon = None
        self.HP = 1 # 1 = 100%
        self.lives = 4
        self.num_caught = 0
        self.reset_stage()
        # Load precalculated resources for choosing Pokemon and moves
        self.boss_pokemon = pickle.load(open(self.boss_pokemon_path, 'rb'))
        self.rental_pokemon = pickle.load(open(self.rental_pokemon_path, 'rb'))
        self.boss_matchups = pickle.load(open(self.boss_matchups_path, 'rb'))
        self.rental_matchups = pickle.load(open(self.rental_matchups_path, 'rb'))
        self.rental_scores = pickle.load(open(self.rental_scores_path, 'rb'))
        
        self.items = pickle.load(open(self.items_path, 'rb'))
        
    def reset_stage(self) -> None:
        """Reset after a battle."""
        self.move_index = 0
        self.dmax_timer = -1
        self.opponent = None
        self.dynamax_available = False
        if self.pokemon is not None:
            if self.pokemon.name == 'Ditto':
                self.pokemon = self.rental_pokemon['Ditto']
            self.pokemon.dynamax = False
        

    def get_frame(self,
                  stage: str='') -> Image:
        """Get an annotated image of the current Switch output."""
        img = self.cap.read()[1]

        # Draw rectangles around detection areas
        h, w = img.shape[:2]
        if stage == 'select_pokemon':
            cv2.rectangle(img, (round(self.shiny_rect[0][0]*w)-2,round(self.shiny_rect[0][1]*h)-2),
                          (round(self.shiny_rect[1][0]*w)+2,round(self.shiny_rect[1][1]*h)+2), (0,255,0), 2)
        elif stage == 'join':
            cv2.rectangle(img, (round(self.sel_rect_1[0][0]*w)-2,round(self.sel_rect_1[0][1]*h)-2),
                          (round(self.sel_rect_1[1][0]*w)+2,round(self.sel_rect_1[1][1]*h)+2), (0,255,0), 2)
            cv2.rectangle(img, (round(self.sel_rect_2[0][0]*w)-2,round(self.sel_rect_2[0][1]*h)-2),
                          (round(self.sel_rect_2[1][0]*w)+2,round(self.sel_rect_2[1][1]*h)+2), (0,255,0), 2)
            cv2.rectangle(img, (round(self.sel_rect_3[0][0]*w)-2,round(self.sel_rect_3[0][1]*h)-2),
                          (round(self.sel_rect_3[1][0]*w)+2,round(self.sel_rect_3[1][1]*h)+2), (0,255,0), 2)
            cv2.rectangle(img, (round(self.abil_rect_1[0][0]*w)-2,round(self.abil_rect_1[0][1]*h)-2),
                          (round(self.abil_rect_1[1][0]*w)+2,round(self.abil_rect_1[1][1]*h)+2), (0,255,255), 2)
            cv2.rectangle(img, (round(self.abil_rect_2[0][0]*w)-2,round(self.abil_rect_2[0][1]*h)-2),
                          (round(self.abil_rect_2[1][0]*w)+2,round(self.abil_rect_2[1][1]*h)+2), (0,255,255), 2)
            cv2.rectangle(img, (round(self.abil_rect_3[0][0]*w)-2,round(self.abil_rect_3[0][1]*h)-2),
                          (round(self.abil_rect_3[1][0]*w)+2,round(self.abil_rect_3[1][1]*h)+2), (0,255,255), 2)
        elif stage == 'catch':
            cv2.rectangle(img, (round(self.sel_rect_4[0][0]*w)-2,round(self.sel_rect_4[0][1]*h)-2),
                          (round(self.sel_rect_4[1][0]*w)+2,round(self.sel_rect_4[1][1]*h)+2), (0,255,0), 2)
            cv2.rectangle(img, (round(self.abil_rect_4[0][0]*w)-2,round(self.abil_rect_4[0][1]*h)-2),
                          (round(self.abil_rect_4[1][0]*w)+2,round(self.abil_rect_4[1][1]*h)+2), (0,255,255), 2)
            cv2.rectangle(img, (round(self.ball_rect[0][0]*w)-2,round(self.ball_rect[0][1]*h)-2),
                          (round(self.ball_rect[1][0]*w)+2,round(self.ball_rect[1][1]*h)+2), (0,0,255), 2)
            cv2.rectangle(img, (round(self.ball_n_rect[0][0]*w)-2,round(self.ball_n_rect[0][1]*h)-2),
                          (round(self.ball_n_rect[1][0]*w)+2,round(self.ball_n_rect[1][1]*h)+2), (0,0,255), 2)
        elif stage == 'battle':
            cv2.rectangle(img, (round(self.sel_rect_5[0][0]*w)-2,round(self.sel_rect_5[0][1]*h)-2),
                          (round(self.sel_rect_5[1][0]*w)+2,round(self.sel_rect_5[1][1]*h)+2), (0,255,0), 2)
            cv2.rectangle(img, (round(self.type_rect_1[0][0]*w)-2,round(self.type_rect_1[0][1]*h)-2),
                          (round(self.type_rect_1[1][0]*w)+2,round(self.type_rect_1[1][1]*h)+2), (255,255,0), 2)
            cv2.rectangle(img, (round(self.type_rect_2[0][0]*w)-2,round(self.type_rect_2[0][1]*h)-2),
                          (round(self.type_rect_2[1][0]*w)+2,round(self.type_rect_2[1][1]*h)+2), (255,255,0), 2)
        elif stage == 'backpacker':
            cv2.rectangle(img, (round(self.item_rect_1[0][0]*w)-2,round(self.item_rect_1[0][1]*h)-2),
                          (round(self.item_rect_1[1][0]*w)+2,round(self.item_rect_1[1][1]*h)+2), (0,255,0), 2)
            cv2.rectangle(img, (round(self.item_rect_2[0][0]*w)-2,round(self.item_rect_2[0][1]*h)-2),
                          (round(self.item_rect_2[1][0]*w)+2,round(self.item_rect_2[1][1]*h)+2), (0,255,0), 2)
            cv2.rectangle(img, (round(self.item_rect_3[0][0]*w)-2,round(self.item_rect_3[0][1]*h)-2),
                          (round(self.item_rect_3[1][0]*w)+2,round(self.item_rect_3[1][1]*h)+2), (0,255,0), 2)
            cv2.rectangle(img, (round(self.item_rect_4[0][0]*w)-2,round(self.item_rect_4[0][1]*h)-2),
                          (round(self.item_rect_4[1][0]*w)+2,round(self.item_rect_4[1][1]*h)+2), (0,255,0), 2)
            cv2.rectangle(img, (round(self.item_rect_5[0][0]*w)-2,round(self.item_rect_5[0][1]*h)-2),
                          (round(self.item_rect_5[1][0]*w)+2,round(self.item_rect_5[1][1]*h)+2), (0,255,0), 2)

        # Return the scaled and annotated image
        return img


    def read_text(self,
                  img: Image,
                  section: Tuple[Tuple[float, float], Tuple[float, float]]=((0,0),(1,1)),
                  threshold: bool=True,
                  invert: bool=False,
                  language: str='spa',
                  segmentation_mode: str='--psm 11') -> str:
        """Read text from a section (default entirety) of an image using Tesseract."""
        # Image is optionally supplied, usually when multiple text areas must be read so the image only needs to be fetched once
        if img is None:
            img = self.get_frame()

        # Process image according to instructions
        h, w, channels = img.shape
        if threshold:
            img = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (0,0,100), (180,15,255))
        if invert:
            img = cv2.bitwise_not(img)
        img = img[round(section[0][1]*h):round(section[1][1]*h),
                  round(section[0][0]*w):round(section[1][0]*w)]
        #cv2.imshow('Text Area', img) # DEBUG

        # Read text using Tesseract and return the raw text
        self.lock.release()
        if self.exit_flag.is_set():
            sys.exit()
        text = pytesseract.image_to_string(img, lang=language, config=segmentation_mode)
        self.lock.acquire()

        return text

    def identify_pokemon(self,
                         name: str,
                         ability: str='',
                         types: str='') -> Pokemon:
        """Match OCRed Pokemon to a rental Pokemon."""
        text = name.replace('\n','')+ability.replace('\n','')+types.replace('\n','')
        matched_text = ''
        best_match = None
        match_value = 1000
        for key in self.rental_pokemon.keys():
            pokemon = self.rental_pokemon[key]
            string_to_match = pokemon.name.split(' (')[0]
            if ability != '':
                string_to_match += pokemon.ability
            if types != '':
                string_to_match += pokemon.types[0] + pokemon.types[1]
            distance = enchant.utils.levenshtein(text, string_to_match)
            if distance < match_value:
                match_value = distance
                best_match = pokemon
                matched_text = string_to_match
        if match_value > len(text)/3:
            print('WARNING: could not find a good match for Pokemon: "'+text+'"')
            pass
        print('OCRed Pokemon:\t'+text+'\nMatched to:\t'+matched_text+' @ '+str(match_value)) # DEBUG
        return best_match

    def read_selectable_pokemon(self,
                                stage: str) -> List[Pokemon]:
        """Return a list of available Pokemon names."""
        # Fetch the image from the Switch output
        image = self.get_frame()

        # Get a list of Pokemon names present, depending on stage
        pokemon_names = []
        abilities = []
        types = []
        pokemon_list = []
        if stage == 'join':
            pokemon_names.append(self.read_text(image, self.sel_rect_1, threshold=False, invert=True, language=None, segmentation_mode='--psm 8').strip())
            pokemon_names.append(self.read_text(image, self.sel_rect_2, threshold=False, language=None, segmentation_mode='--psm 8').strip())
            pokemon_names.append(self.read_text(image, self.sel_rect_3, threshold=False, language=None, segmentation_mode='--psm 3').strip()) # This last name shifts around between runs necessitating a bigger rectangle and different text segmentation mode
            abilities.append(self.read_text(image, self.abil_rect_1, threshold=False, invert=True, language=None, segmentation_mode='--psm 8').strip())
            abilities.append(self.read_text(image, self.abil_rect_2, threshold=False, language=None, segmentation_mode='--psm 8').strip())
            abilities.append(self.read_text(image, self.abil_rect_3, threshold=False, language=None, segmentation_mode='--psm 3').strip())
            types = ['','','']
        elif stage == 'catch':
            pokemon_names.append(self.read_text(image, self.sel_rect_4, threshold=False, language=None, segmentation_mode='--psm 3').strip().split('\n')[-1])
            abilities.append(self.read_text(image, self.abil_rect_4, threshold=False, language=None, segmentation_mode='--psm 3').strip())
            types.append('')
        elif stage == 'battle':
            pokemon_names.append(self.read_text(image, self.sel_rect_5, threshold=False, invert=False, segmentation_mode='--psm 8').strip())
            abilities.append('')
            type_1 = self.read_text(image, self.type_rect_1, threshold=False, invert=True, segmentation_mode='--psm 8').strip().title()
            type_2 = self.read_text(image, self.type_rect_2, threshold=False, invert=True, segmentation_mode='--psm 8').strip().title()
            types.append(type_1+type_2)

        # Identify the Pokemon based on its name and ability/types, where relevant
        for i in range(len(pokemon_names)):
            pokemon_list.append(self.identify_pokemon(pokemon_names[i], abilities[i], types[i]))

        # Return the list of Pokemon
        return pokemon_list
    
    def check_shiny(self) -> bool:
        """Detect whether a Pokemon is shiny by looking for the icon in the summary screen."""
        # Fetch, crop, and threshold image so the red shiny star will appear white and everything else appears black
        img = cv2.cvtColor(self.get_frame(), cv2.COLOR_BGR2HSV)
        h, w, channels = img.shape
        shiny_area = img[round(self.shiny_rect[0][1]*h):round(self.shiny_rect[1][1]*h),
                         round(self.shiny_rect[0][0]*w):round(self.shiny_rect[1][0]*w)]
        # Measure the average value in the shiny star area
        measured_value = cv2.inRange(shiny_area, (0,100,0), (180,255,255)).mean()
        # The shiny star results in a measured_value even greater than 10
        return True if measured_value > 1 else False

    def get_target_ball(self) -> str:
        """Return the name of the Poke Ball needed."""
        return self.base_ball if self.num_caught < 3 else self.legendary_ball

    def check_ball(self) -> str:
        """Detect the currently selected Poke Ball during the catch phase of the game."""
        return self.read_text(self.get_frame(), self.ball_rect, threshold=False, invert=True, language='spa', segmentation_mode='--psm 8').strip()
        
    def record_ball_use(self) -> None:
        """Decrement the number of balls in the inventory and increment the number of pokemon caught."""
        if self.base_ball == self.legendary_ball:
            self.base_balls -= 1
            self.legendary_balls -= 1
        elif self.num_caught < 3:
            self.base_balls -= 1
        else:
            self.legendary_balls -= 1
        self.num_caught += 1
        self.caught_total += 1

    def check_sufficient_balls(self) -> bool:
        """Calculate whether sufficient balls remain for another run."""
        return False if (self.base_ball == self.legendary_ball and self.base_balls < 4) or (self.base_balls < 3) or (self.legendary_balls < 1) else True

    def calculate_ore_cost(self, num_resets: int) -> int:
        """Calculate the prospective Dynite Ore cost of resetting the game."""
        return 0 if num_resets < 3 else min(10, num_resets)

    def check_sufficient_ore(self, num_resets: int) -> bool:
        """Calculate whether sufficient Dynite Ore remains to quit the run without saving."""
        return True if self.dynite_ore >= self.calculate_ore_cost(num_resets) else False

    def push_buttons(self, *commands: Tuple[str, float]) -> None:
        """Send messages to the microcontroller telling it to press buttons on the Switch."""
        for character, duration in commands:
            self.lock.release()
            if self.exit_flag.is_set():
                sys.exit()
            self.com.write(character)
            time.sleep(duration)
            self.lock.acquire()

    def log(self,
            string: str='') -> None:
        """Print a string to the log file with a timestamp."""
        with open(self.filename, 'a') as file:
            file.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'\t'+string+'\n')
        print(string)
        
    def display_results(self, log=False, screenshot=False):
        """Display video from the Switch alongside some annotations describing the run sequence."""
        # Calculate some statistics for display        
        win_percent = 0 if self.runs == 0 else 100 * self.wins / self.runs
        time_per_run = 0 if self.runs == 0 else (datetime.now() - self.start_date) / self.runs
        shiny_percent = 0 if self.shinies_found == 0 else 100 * self.shinies_found / self.caught_total

        # Expand the image with blank space for writing results
        frame = cv2.copyMakeBorder(self.get_frame(stage=self.stage), 0, 0, 0, 200, cv2.BORDER_CONSTANT)
        width = frame.shape[1]

        # Construct arrays of text and values to display
        labels = ('Run #',
                  'Stage: ',
                  'Battle: ',
                  'Lives: ',
                  'Pokemon: ',
                  'Opponent: ',
                  'Wins: ',
                  'Time per run: ',
                  'Pokemon caught: ',
                  'Shinies found: ',
                  'Base balls: ',
                  'Boss balls: ',
                  'Mode: ',
                  'Dynite Ore: ')
        values = (str(self.runs + 1)+' : '+str(self.boss),
                  self.stage,
                  str(self.num_caught + 1),
                  str(self.lives),
                  str(self.pokemon),
                  str(self.opponent),
                  str(self.wins)+' (%0.2f'%win_percent+' %)',
                  str(time_per_run),
                  str(self.caught_total),
                  str(self.shinies_found)+' (%0.2f'%shiny_percent+' %)',
                  str(self.base_ball)+' : '+str(self.base_balls),
                  str(self.legendary_ball)+' : '+str(self.legendary_balls),
                  str(self.mode),
                  str(self.dynite_ore))


        for i in range(len(labels)):
            cv2.putText(frame, labels[i] + values[i], (width - 195, 25 + 25 * i), cv2.FONT_HERSHEY_PLAIN, 0.8,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if log:
                self.log(labels[i] + values[i])

        # Display
        cv2.imshow('Output', frame)
        if log or screenshot:
            # Save a copy of the final image
            cv2.imwrite(self.filename[:-8] + '_cap_'+str(self.shot)+'.png', frame)
            self.shot += 1
            screenshot = False

# WIP Backpacker item selection

    def read_selectable_item(self) -> List[str]:
        """Return a list of available Pokemon names."""
        # Fetch the image from the Switch output
        image = self.get_frame()

        # Get a list of Pokemon names present, depending on stage
        items = []
        itlist = []
        items.append(self.read_text(image, self.item_rect_1, threshold=False, invert=True, language=None, segmentation_mode='--psm 8').strip())
        items.append(self.read_text(image, self.item_rect_2, threshold=False, language=None, segmentation_mode='--psm 8').strip())
        items.append(self.read_text(image, self.item_rect_3, threshold=False, language=None, segmentation_mode='--psm 8').strip())
        items.append(self.read_text(image, self.item_rect_4, threshold=False, language=None, segmentation_mode='--psm 8').strip())
        items.append(self.read_text(image, self.item_rect_5, threshold=False, language=None, segmentation_mode='--psm 8').strip())

        # Identify the Pokemon based on its name and ability/types, where relevant
        for i in range(len(items)):
            itlist.append(self.identify_item(items[i]))
            #item_list.append(items[i])
            #print(str(items[i]))

        # Return the list of Pokemon
        return itlist

    def identify_item(self,
                         item: str) -> str:
        """Match OCRed items to a hold item."""
        text = item.replace('\n','')
        best_match = None
        match_value = 1000
        for i in range(len(self.items)):
            item_to_match = str(self.items[i])
            distance = enchant.utils.levenshtein(text, item_to_match)
            if distance < match_value:
                match_value = distance
                best_match = item_to_match
        if match_value > len(text):
            print('WARNING: could not find a good match for item: "'+text+'"')
            pass
        #print('OCRed item:\t'+text+'\nMatched to:\t'+best_match+' @ '+str(match_value)) # DEBUG
        print(str(best_match))
        return best_match

    def value_items(self, listed_items) -> int:
        selection = 0
        val = 0
        values = [0,0,0,0,0]
        #Check for desired items on each slot
        for i in range(len(listed_items)):
            for j in range(len(self.tier_5_items)):
                if self.tier_5_items[j] in listed_items[i]:
                    values[i] = 1
            for j in range(len(self.tier_4_items)):
                if self.tier_4_items[j] in listed_items[i]:
                    values[i] = 2
            for j in range(len(self.tier_3_items)):
                if self.tier_3_items[j] in listed_items[i]:
                    values[i] = 3
            for j in range(len(self.tier_2_items)):
                if self.tier_2_items[j] in listed_items[i]:
                    values[i] = 4
            for j in range(len(self.tier_1_items)):
                if self.tier_1_items[j] in listed_items[i]:
                    values[i] = 5
        #Look for the best valued item            
        for i in range(len(values)):
            if val < values[i]:
                val = values[i]
                selection = i
        
        return selection

    def pokeball_selection(self) -> str:
        self.base_ball = 'DEFAULT'
        for i in range(len(self.base_dive)):
                if str(self.opponent) in self.base_dive[i]:
                    self.base_ball = 'Buceo'
        for i in range(len(self.base_lux)):
                if str(self.opponent) in self.base_lux[i]:
                    self.base_ball = 'Lujo'
        for i in range(len(self.base_premier)):
                if str(self.opponent) in self.base_premier[i]:
                    self.base_ball = 'Honor'
        for i in range(len(self.base_dream)):
                if str(self.opponent) in self.base_dream[i]:
                    self.base_ball = 'Ensue'
    

        
