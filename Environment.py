from Hand import Hand
from Deck import Deck
from Action import Action


class Environment:
    def __init__(self, number_decks, epochs):
        self.number_decks = number_decks
        self.player_hand = Hand()
        self.dealer_hand = Hand()

        self.number_games = epochs
        self.wins = 0
        self.loses = 0
        self.draws = 0

        self.win_track = []
        self.lose_track = []
        self.draw_track = []

    def get_result(self):
        return "win rate: "+ str((self.wins/self.number_games)*100)  +"%\nlose rate: " + str((self.loses/self.number_games)*100) + "%\nwins: " + str(self.wins) + "\nloses: " + str(self.loses) + "\ndraws: " + str(self.draws)

    def reset(self):
        self.deck = Deck(self.number_decks)
        self.deck.shuffle()
        
        self.player_hand.reset()
        self.dealer_hand.reset()
        self._init_deal()

        return self._get_state()
        
    def _init_deal(self):
        self.player_hand.add_card(self.deck.deal())
        self.dealer_hand.add_card(self.deck.deal())
        self.player_hand.add_card(self.deck.deal())
        self.dealer_hand.add_card(self.deck.deal())

    def _is_bust(self, value):
        return value > 21

    def step(self, action):
        bust = False
        terminate = False
        reward = 0
        if action == Action.Hit:
            self.player_hand.add_card(self.deck.deal())
            bust = self._is_bust(self.player_hand.get_value()) #check if bust, reward should still be 0?
            if bust:
                _, reward = self._evaluate() #must increase lose.
        else:
            self._finish_dealer()
            bust, reward = self._evaluate()
            terminate = True

        return (self._get_state(), reward, bust, terminate)

    def get_track_lists(self):
        return (self.win_track, self.lose_track, self.draw_track)

    def _finish_dealer(self):
        while self.dealer_hand.get_value() < 17: #according to the dealer rules, must draw until higher than 16 and then stand.
            self.dealer_hand.add_card(self.deck.deal())

    def _evaluate(self):
        player_value = self.player_hand.get_value()
        dealer_value = self.dealer_hand.get_value()

        ##print("Epoch Result:\nplayer: " + str(self.player_hand) + "\ndealer: " + str(self.dealer_hand))
        
        
        #returns bust, reward.
        if self._is_bust(player_value):
            self.loses += 1
            self._update_tracking()
            #print("lose, player bust.")
            return True, -1 #player bust, lose.
        elif self._is_bust(dealer_value):
            self.wins += 1
            self._update_tracking()
            #print("win, dealer bust.")
            return False, 1 #dealer bust, but player did not, still win.
        elif dealer_value > player_value:
            self.loses += 1
            self._update_tracking()
            #print("lose, dealer closer to 21.")
            return False, -1 #player not bust but less than dealer, also lose.
        elif player_value > dealer_value:
            self.wins += 1
            self._update_tracking()
            #print("win, player closer to 21.")
            return False, 1 #no one bust, player has higher value, win. 
        #print("draw, both have " + str(player_value) + ".")
        self.draws += 1
        self._update_tracking()
        return False, 0 #it's a draw, neutral reward.

    def _update_tracking(self):
        self.lose_track.append(self.loses)
        self.draw_track.append(self.draws)
        self.win_track.append(self.wins)


    def _get_state(self):
        return (self.player_hand.get_value(), self.dealer_hand.get_first_value(), self.player_hand.has_soft_ace())


